"""Numba port of the Bullet-trained Chess768 network (768 -> 1024x2 -> 8 buckets, SCReLU),
matching search.py's existing NNUE interface exactly (nnue_init_root, nnue_make_move,
nnue_make_null, nnue_eval, new_accumulator_stack) so search.py can use this network with NO
changes -- just point agent.py at this module instead of the HalfKAv2_hm one. This makes the
eventual comparison a clean "same search, different network" test.

One real simplification versus the HalfKAv2_hm version: Chess768 features don't depend on king
position at all, so there's no king-moved-triggers-full-refresh branch needed here -- every
move, including king moves, uses the fast incremental diff path.

`psqt` is threaded through purely for interface compatibility with search.py's calls -- this
network has no PSQT buckets at all (output bucketing here is by material count, computed fresh
from piece_count every eval, not stored per-ply), so it's allocated as a tiny unused dummy array
and never read or written meaningfully.
"""

import os

import numpy as np
from numba import njit

from board import KING, bit, lsb, popcount

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bullet_raw.npz")
NNUE_AVAILABLE = os.path.exists(_WEIGHTS_PATH)

NNUE_STACK_SIZE = 128
L0_HIDDEN = 1024
NUM_INPUTS = 768
NUM_BUCKETS = 8
L1_INPUT = 2 * L0_HIDDEN
BUCKET_DIVISOR = (32 + NUM_BUCKETS - 1) // NUM_BUCKETS  # = 4

if NNUE_AVAILABLE:
    _npz = np.load(_WEIGHTS_PATH)
    FT_WEIGHT = _npz["l0w"].astype(np.float32)  # (768, 1024)
    FT_BIAS = _npz["l0b"].astype(np.float32)  # (1024,)
    # Variant C (empirically the best match found so far -- exact startpos match against
    # Bullet's real reference value): l1w needs swap+transpose, not the direct reshape
    # bullet_network_loader.py originally applied when this .npz was created. Recovering the
    # original flat byte order via .ravel() is safe here since a direct reshape never
    # reorders bytes -- only re-chunks them -- so this is equivalent to reshaping the raw
    # file's original flat sequence directly as (2048, 8) and transposing.
    L1_WEIGHT = _npz["l1w"].astype(np.float32).ravel().reshape(2048, 8).T  # (8, 2048), swap+T
    L1_BIAS = _npz["l1b"].astype(np.float32)  # (8,)
else:
    FT_WEIGHT = np.zeros((1, 1), dtype=np.float32)
    FT_BIAS = np.zeros(1, dtype=np.float32)
    L1_WEIGHT = np.zeros((1, 1), dtype=np.float32)
    L1_BIAS = np.zeros(1, dtype=np.float32)


@njit(cache=False, inline="always")
def _feature_index(is_white_pov, sq, piece_type, piece_is_white):  # type: ignore[no-untyped-def]
    """TESTING Variant C: friendly-first (0-5), enemy-second (6-11), combined with the
    swap+transpose l1w layout above. This combination gave an EXACT match to Bullet's real
    startpos value (0.1236) -- the strongest single piece of evidence found so far -- though
    its queen-ablation delta (-3.63) doesn't yet match Bullet's real -2.51 as closely. Testing
    further before fully committing to this over the previous enemy-first/direct-l1w version.
    """
    friendly = piece_is_white if is_white_pov else not piece_is_white
    base = (piece_type - 1) if friendly else (piece_type - 1 + 6)
    used_sq = sq if is_white_pov else (sq ^ 56)
    return 64 * base + used_sq


@njit(cache=False)
def _piece_type_at(bb, sq):  # type: ignore[no-untyped-def]
    b = bit(sq)
    for pt in range(1, 7):
        if bb[pt + 1] & b:
            return pt
    return 0


@njit(cache=False)
def refresh_accumulator(bb, is_white_pov, acc_out):  # type: ignore[no-untyped-def]
    acc_out[:] = FT_BIAS
    occ = bb[0] | bb[1]
    while occ:
        sq = lsb(occ)
        occ &= occ - 1
        piece_type = _piece_type_at(bb, sq)
        piece_is_white = (bb[0] & bit(sq)) != 0
        idx = _feature_index(is_white_pov, sq, piece_type, piece_is_white)
        acc_out += FT_WEIGHT[idx]


@njit(cache=False)
def nnue_init_root(bb, acc, psqt):  # type: ignore[no-untyped-def]
    refresh_accumulator(bb, True, acc[0, 0])
    refresh_accumulator(bb, False, acc[0, 1])


@njit(cache=False)
def nnue_make_move(bbs, mbs, acc, psqt, ply):  # type: ignore[no-untyped-def]
    """Diffs the mailbox at ply vs ply+1, exactly the same generic approach as the
    HalfKAv2_hm version -- correctly handles quiet moves, captures, en passant, castling, and
    promotion uniformly, since all of them are fully described by "which squares changed"."""
    acc[ply + 1, 0] = acc[ply, 0]
    acc[ply + 1, 1] = acc[ply, 1]
    for sq in range(64):
        before_piece = mbs[ply, sq]
        after_piece = mbs[ply + 1, sq]
        before_white = (bbs[ply, 0] & bit(sq)) != 0
        after_white = (bbs[ply + 1, 0] & bit(sq)) != 0
        if before_piece == after_piece and (before_piece == 0 or before_white == after_white):
            continue
        if before_piece != 0:
            w_idx = _feature_index(True, sq, before_piece, before_white)
            b_idx = _feature_index(False, sq, before_piece, before_white)
            acc[ply + 1, 0] -= FT_WEIGHT[w_idx]
            acc[ply + 1, 1] -= FT_WEIGHT[b_idx]
        if after_piece != 0:
            w_idx = _feature_index(True, sq, after_piece, after_white)
            b_idx = _feature_index(False, sq, after_piece, after_white)
            acc[ply + 1, 0] += FT_WEIGHT[w_idx]
            acc[ply + 1, 1] += FT_WEIGHT[b_idx]


@njit(cache=False)
def nnue_make_null(acc, psqt, ply):  # type: ignore[no-untyped-def]
    acc[ply + 1, 0] = acc[ply, 0]
    acc[ply + 1, 1] = acc[ply, 1]


@njit(cache=False)
def _screlu(x):  # type: ignore[no-untyped-def]
    clipped = np.minimum(np.maximum(x, 0.0), 1.0)
    return clipped * clipped


@njit(cache=False)
def forward(white_acc, black_acc, side, piece_count):  # type: ignore[no-untyped-def]
    stm_acc = white_acc if side == 0 else black_acc
    ntm_acc = black_acc if side == 0 else white_acc

    hidden = np.empty(L1_INPUT, dtype=np.float32)
    hidden[:L0_HIDDEN] = _screlu(stm_acc)
    hidden[L0_HIDDEN:] = _screlu(ntm_acc)

    bucket = (piece_count - 2) // BUCKET_DIVISOR
    if bucket < 0:
        bucket = 0
    if bucket > NUM_BUCKETS - 1:
        bucket = NUM_BUCKETS - 1

    result = np.float32(0.0)
    for i in range(L1_INPUT):
        result += L1_WEIGHT[bucket, i] * hidden[i]
    return result + L1_BIAS[bucket]


NNUE2SCORE = 400.0  # confirmed directly from chessathon_s1.rs's `eval_scale: 400.0` -- this
                     # is the real, training-time calibration for this network, not borrowed
                     # from the other (HalfKAv2) network's convention


@njit(cache=False)
def nnue_eval(acc, psqt, side, piece_count, ply):  # type: ignore[no-untyped-def]
    raw = forward(acc[ply, 0], acc[ply, 1], side, piece_count)
    return np.int64(raw * NNUE2SCORE)


def new_accumulator_stack():
    """psqt is a tiny unused dummy here, kept only so the (acc, psqt) tuple shape matches
    what agent.py already expects to unpack and pass around."""
    if NNUE_AVAILABLE:
        acc = np.zeros((NNUE_STACK_SIZE, 2, L0_HIDDEN), dtype=np.float32)
        psqt = np.zeros((NNUE_STACK_SIZE, 2, 1), dtype=np.float32)
    else:
        acc = np.zeros((1, 2, 1), dtype=np.float32)
        psqt = np.zeros((1, 2, 1), dtype=np.float32)
    return acc, psqt
