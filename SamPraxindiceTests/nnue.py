"""NNUE evaluator: numba port of nnue_eval.py / nnue_accumulator.py, operating directly on the
bb/st arrays from board.py rather than python-chess objects, matching how every other hot-path
function in this engine works. Every formula here is the same one validated in nnue_eval.py
against the real trained model (exact match) and nnue_accumulator.py against full recompute
(float32-precision match across 2,400+ simulated updates plus targeted edge cases) -- this file
only changes *how* that logic is expressed (numba/bb-arrays instead of numpy/python-chess), not
what it computes.

Weights load from "weights.npz" next to this file at import time. If that file isn't present,
NNUE_AVAILABLE is False and callers should fall back to evaluate.py's hand-crafted eval.
"""

import math
import os

import numpy as np
from numba import njit

from board import KING, PAWN, bit, lsb, popcount

# fmt: off
KING_BUCKETS = np.array([
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4,
    -1, -1, -1, -1, 3, 2, 1, 0,
], dtype=np.int64)
# fmt: on

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")
NNUE_AVAILABLE = os.path.exists(_WEIGHTS_PATH)

if NNUE_AVAILABLE:
    _npz = np.load(_WEIGHTS_PATH)
    # The feature transformer ships as the integers the network was trained against
    # (nnue-pytorch: weights and biases x256 as int16, psqt x9600 as int32), so the accumulator
    # is maintained in int16 exactly as Stockfish does: half the bytes per row of fp32 and twice
    # the lanes per vector add. An accumulator is the bias plus at most 32 rows (33 transiently
    # inside a diff); the guard checks that even the 33 largest weights of every column cannot
    # leave int16, since numba integer overflow is silent.
    FT_WEIGHT = np.ascontiguousarray(_npz["ft_weight"], dtype=np.int16)
    FT_PSQT_WEIGHT = np.ascontiguousarray(_npz["ft_psqt_weight"], dtype=np.int32)
    FT_BIAS = np.ascontiguousarray(_npz["ft_bias"], dtype=np.int16)
    _absw = np.abs(FT_WEIGHT)
    _bound = np.partition(_absw, _absw.shape[0] - 33, axis=0)[-33:].astype(np.int32).sum(axis=0)
    _bound += np.abs(FT_BIAS).astype(np.int32)
    if int(_bound.max()) >= 32767:
        raise RuntimeError(f"int16 accumulator could overflow: column bound {int(_bound.max())}")
    del _absw, _bound
    # The layer stacks ship already quantized: int8 weights and int32 biases on the scales the
    # network was trained against (nnue-pytorch: L1 weight x128, L2 x64, output x128, each
    # bias x weight-scale x128). They are held as float32 carrying those integer values:
    # every product and partial sum in forward() is an integer below 2**24, so float32
    # arithmetic on them is exact, and it is the arithmetic LLVM vectorises well. L1 is
    # transposed to (bucket, input, output) so the 32 weights of one input are contiguous.
    LS_L1_WT = np.ascontiguousarray(_npz["ls_l1_weight"].transpose(0, 2, 1)).astype(np.float32)
    LS_L1_BQ = _npz["ls_l1_bias"].astype(np.int64)
    LS_L2_WQ = np.ascontiguousarray(_npz["ls_l2_weight"]).astype(np.float32)
    LS_L2_BQ = _npz["ls_l2_bias"].astype(np.int64)
    LS_OUT_WQ = np.ascontiguousarray(_npz["ls_out_weight"][:, 0, :]).astype(np.float32)
    LS_OUT_BQ = _npz["ls_out_bias"][:, 0].astype(np.int64)
    L1 = int(_npz["meta_L1"])
    NUM_PSQT_BUCKETS = int(_npz["meta_num_psqt_buckets"])
else:
    # placeholder zero-size arrays so the module still imports cleanly and njit functions below
    # still compile (numba needs concrete array types even if they're never actually reached)
    FT_WEIGHT = np.zeros((1, 1024), dtype=np.int16)
    FT_PSQT_WEIGHT = np.zeros((1, 8), dtype=np.int32)
    FT_BIAS = np.zeros(1024, dtype=np.int16)
    LS_L1_WT = np.zeros((8, 1024, 32), dtype=np.float32)
    LS_L1_BQ = np.zeros((8, 32), dtype=np.int64)
    LS_L2_WQ = np.zeros((8, 32, 64), dtype=np.float32)
    LS_L2_BQ = np.zeros((8, 32), dtype=np.int64)
    LS_OUT_WQ = np.zeros((8, 128), dtype=np.float32)
    LS_OUT_BQ = np.zeros(8, dtype=np.int64)
    L1 = 1024
    NUM_PSQT_BUCKETS = 8


NNUE_STACK_SIZE = 128  # safe upper bound over search.py's actual STACK (104); avoids a
                        # circular import just to match it exactly


@njit(cache=False, fastmath=True)
def nnue_init_root(bb, acc, psqt):  # type: ignore[no-untyped-def]
    """Call once, at the very start of a search, to populate ply 0's accumulators from
    scratch. `acc` is shaped (STACK, 2, L1), `psqt` (STACK, 2, num_psqt_buckets) -- both
    allocated by the caller (agent.py) and passed in explicitly, the same way bbs/sts/mbs
    already are, since numba freezes module-level array globals as read-only inside njit
    functions and there is no way to mutate one from here otherwise."""
    refresh_accumulator(bb, True, acc[0, 0], psqt[0, 0])
    refresh_accumulator(bb, False, acc[0, 1], psqt[0, 1])


@njit(cache=False, fastmath=True)
def nnue_make_move(bbs, mbs, acc, psqt, ply):  # type: ignore[no-untyped-def]
    """Call immediately AFTER board.make_move(bbs, sts, mbs, ply, move) has written ply+1.
    Diffs the mailbox at ply vs ply+1 to find every square that changed, rather than
    interpreting the move's own encoding -- this correctly and generically handles quiet
    moves, captures, en passant, castling, and promotion with the same code path, since all of
    them are fully described by "which squares changed piece" regardless of *why*."""
    white_king_before = bbs[ply, 7] & bbs[ply, 0]
    white_king_after = bbs[ply + 1, 7] & bbs[ply + 1, 0]
    black_king_before = bbs[ply, 7] & bbs[ply, 1]
    black_king_after = bbs[ply + 1, 7] & bbs[ply + 1, 1]

    if white_king_before != white_king_after:
        refresh_accumulator(bbs[ply + 1], True, acc[ply + 1, 0], psqt[ply + 1, 0])
    else:
        acc[ply + 1, 0] = acc[ply, 0]
        psqt[ply + 1, 0] = psqt[ply, 0]
        _apply_diff(bbs, mbs, ply, True, acc[ply + 1, 0], psqt[ply + 1, 0])

    if black_king_before != black_king_after:
        refresh_accumulator(bbs[ply + 1], False, acc[ply + 1, 1], psqt[ply + 1, 1])
    else:
        acc[ply + 1, 1] = acc[ply, 1]
        psqt[ply + 1, 1] = psqt[ply, 1]
        _apply_diff(bbs, mbs, ply, False, acc[ply + 1, 1], psqt[ply + 1, 1])


@njit(cache=False, fastmath=True)
def _apply_diff(bbs, mbs, ply, is_white_pov, acc_out, psqt_out):  # type: ignore[no-untyped-def]
    pov_colour = 0 if is_white_pov else 1
    king_sq = lsb(bbs[ply, 7] & bbs[ply, pov_colour])
    # Iterate only over squares that actually changed. XORing all 8 bitboard planes (white,
    # black, and the 6 piece-type planes) between ply and ply+1 gives a bitboard where any
    # square whose contents changed has at least one bit set. This covers piece add/remove
    # (colour bit flips), promotion (piece-type bit changes), and same-type capture (colour
    # bits flip). Only 2 to 4 squares change on a typical move, so this replaces ~60 wasted
    # branchy iterations of a full 64-square scan with a few lsb-driven ones.
    changed = np.int64(0)
    for plane in range(8):
        changed |= bbs[ply, plane] ^ bbs[ply + 1, plane]
    while changed:
        sq = lsb(changed)
        changed &= changed - 1
        before_piece = mbs[ply, sq]
        after_piece = mbs[ply + 1, sq]
        before_white = (bbs[ply, 0] & bit(sq)) != 0
        after_white = (bbs[ply + 1, 0] & bit(sq)) != 0
        if before_piece != 0:
            idx = perspective_feature(is_white_pov, sq, before_piece, before_white, king_sq)
            acc_out -= FT_WEIGHT[idx]
            psqt_out -= FT_PSQT_WEIGHT[idx]
        if after_piece != 0:
            idx = perspective_feature(is_white_pov, sq, after_piece, after_white, king_sq)
            acc_out += FT_WEIGHT[idx]
            psqt_out += FT_PSQT_WEIGHT[idx]


@njit(cache=False, fastmath=True)
def nnue_make_null(acc, psqt, ply):  # type: ignore[no-untyped-def]
    """Null move: no piece placement changes, so both perspectives' accumulators carry over
    unchanged."""
    acc[ply + 1, 0] = acc[ply, 0]
    acc[ply + 1, 1] = acc[ply, 1]
    psqt[ply + 1, 0] = psqt[ply, 0]
    psqt[ply + 1, 1] = psqt[ply, 1]


@njit(cache=False, fastmath=True)
def nnue_eval(acc, psqt, side, piece_count, ply):  # type: ignore[no-untyped-def]
    """Reads this ply's already-maintained accumulators (via nnue_make_move/nnue_make_null)
    and runs the forward pass. Does not touch the board at all -- purely a function of
    whatever's already sitting in acc/psqt[ply]. Returns the evaluation in centipawns as an
    int64, from the side to move's point of view."""
    return forward(acc[ply, 0], acc[ply, 1], psqt[ply, 0], psqt[ply, 1], side, piece_count)


def new_accumulator_stack():
    """Allocates a fresh (acc, psqt) pair sized for one full search stack. Call once per
    independent search context (agent.py needs a separate pair for the main search and for the
    pondering thread, exactly like PONDER_BBS is separate from BBS)."""
    if NNUE_AVAILABLE:
        acc = np.zeros((NNUE_STACK_SIZE, 2, L1), dtype=np.int16)
        psqt = np.zeros((NNUE_STACK_SIZE, 2, NUM_PSQT_BUCKETS), dtype=np.int32)
    else:
        acc = np.zeros((NNUE_STACK_SIZE, 2, 1), dtype=np.int16)
        psqt = np.zeros((NNUE_STACK_SIZE, 2, 1), dtype=np.int32)
    return acc, psqt


@njit(cache=False, fastmath=True)
def orient(is_white_pov, sq, ksq):  # type: ignore[no-untyped-def]
    kfile = ksq & 7
    flip_h = 7 if kfile < 4 else 0
    flip_v = 0 if is_white_pov else 56
    return (flip_h ^ flip_v) ^ sq


@njit(cache=False, fastmath=True)
def perspective_feature(is_white_pov, sq, piece_type, piece_is_white, king_sq):  # type: ignore[no-untyped-def]
    """Feature row index for one piece, from one perspective. `piece_is_white`: True/False.
    `is_white_pov`: which perspective's accumulator this contributes to."""
    pov_is_white = is_white_pov
    o_ksq = orient(is_white_pov, king_sq, king_sq)
    bucket = KING_BUCKETS[o_ksq]
    base = bucket * 704
    if piece_type == KING:
        if piece_is_white == pov_is_white:
            return base + 640 + o_ksq
        else:
            osq = orient(is_white_pov, sq, king_sq)
            return base + 640 + osq
    own = 1 if piece_is_white == pov_is_white else 0
    p_idx10 = (piece_type - 1) * 2 + (1 - own)
    osq = orient(is_white_pov, sq, king_sq)
    return base + p_idx10 * 64 + osq


@njit(cache=False, fastmath=True)
def refresh_accumulator(bb, is_white_pov, acc_out, psqt_out):  # type: ignore[no-untyped-def]
    """Full recompute of one perspective's raw accumulator + psqt vector from a bb array."""
    pov_colour = 0 if is_white_pov else 1
    king_sq = lsb(bb[7] & bb[pov_colour])

    acc_out[:] = FT_BIAS
    psqt_out[:] = 0

    for colour in range(2):
        piece_is_white = colour == 0
        for piece_type in range(1, 7):
            pieces = bb[piece_type + 1] & bb[colour]
            while pieces:
                sq = lsb(pieces)
                pieces &= pieces - 1
                idx = perspective_feature(is_white_pov, sq, piece_type, piece_is_white, king_sq)
                acc_out += FT_WEIGHT[idx]
                psqt_out += FT_PSQT_WEIGHT[idx]


_Q = L1 // 2
_F0 = np.float32(0.0); _F255 = np.float32(255.0)
_INV512 = np.float32(1.0 / 512.0)


@njit(cache=False, fastmath=True)
def forward(white_acc, black_acc, white_psqt, black_psqt, side, piece_count):  # type: ignore[no-untyped-def]
    """Forward pass with the integer semantics the network was trained under (nnue-pytorch's
    fake quantization): activations live on fixed grids and are floored onto them at every
    layer, exactly as Stockfish's integer inference does. Units: the accumulator is already
    int16 on the 1/256 grid, hidden activations to the 1/128 grid, L1 sums are in 1/16384, L2 sums in
    1/8192, the output in 1/16384; x600 converts to centipawns. All arithmetic is on
    integer-valued float32 with every partial sum below 2**24, so it never rounds. L1 is
    evaluated only over the nonzero inputs (typically ~15% of them), eight per iteration.
    `side`: 0 = white to move, 1 = black to move. Returns int64 centipawns."""
    if side == 0:
        own = white_acc; opp = black_acc; us = 1.0
    else:
        own = black_acc; opp = white_acc; us = 0.0
    l0 = np.empty(L1, dtype=np.float32)
    for i in range(_Q):
        a0 = max(min(np.float32(own[i]), _F255), _F0)
        a1 = max(min(np.float32(own[_Q + i]), _F255), _F0)
        l0[i] = np.float32(np.int32(a0 * a1 * _INV512))
        c0 = max(min(np.float32(opp[i]), _F255), _F0)
        c1 = max(min(np.float32(opp[_Q + i]), _F255), _F0)
        l0[_Q + i] = np.float32(np.int32(c0 * c1 * _INV512))
    nz = np.empty(L1, dtype=np.int32)
    n = 0
    for i in range(L1):
        nz[n] = i
        n += (l0[i] != _F0)

    bucket = (piece_count - 1) // 4
    if bucket >= NUM_PSQT_BUCKETS:
        bucket = NUM_PSQT_BUCKETS - 1  # defensive clamp; shouldn't trigger in real chess
    wt = LS_L1_WT[bucket]
    acc = np.zeros(32, dtype=np.float32)
    k = 0
    while k + 7 < n:
        j0 = nz[k]; x0 = l0[j0]; j1 = nz[k + 1]; x1 = l0[j1]; j2 = nz[k + 2]; x2 = l0[j2]; j3 = nz[k + 3]; x3 = l0[j3]
        j4 = nz[k + 4]; x4 = l0[j4]; j5 = nz[k + 5]; x5 = l0[j5]; j6 = nz[k + 6]; x6 = l0[j6]; j7 = nz[k + 7]; x7 = l0[j7]
        for i in range(32):
            acc[i] += (wt[j0, i] * x0 + wt[j1, i] * x1 + wt[j2, i] * x2 + wt[j3, i] * x3
                       + wt[j4, i] * x4 + wt[j5, i] * x5 + wt[j6, i] * x6 + wt[j7, i] * x7)
        k += 8
    while k < n:
        j0 = nz[k]; x0 = l0[j0]
        for i in range(32):
            acc[i] += wt[j0, i] * x0
        k += 1
    l1c = np.empty(32, dtype=np.int64)
    for i in range(32):
        l1c[i] = np.int64(acc[i]) + LS_L1_BQ[bucket, i]
    skip = l1c[30] - l1c[31]

    l1x = np.empty(64, dtype=np.float32)
    for i in range(32):
        t = min(max(l1c[i], -16384), 16384)
        l1x[i] = np.float32(min((t * t) >> 21, 127))
        l1x[32 + i] = np.float32(min(max(l1c[i] >> 7, 0), 127))
    w2 = LS_L2_WQ[bucket]
    l2c = np.empty(32, dtype=np.int64)
    for i in range(32):
        s = _F0
        for j in range(64):
            s += w2[i, j] * l1x[j]
        l2c[i] = np.int64(s) + LS_L2_BQ[bucket, i]
    l2x = np.empty(64, dtype=np.float32)
    for i in range(32):
        t = min(max(l2c[i], -8192), 8192)
        l2x[i] = np.float32(min((t * t) >> 19, 127))
        l2x[32 + i] = np.float32(min(max(l2c[i] >> 6, 0), 127))
    wo = LS_OUT_WQ[bucket]
    s = _F0
    for j in range(64):
        s += wo[j] * l1x[j]
    for j in range(64):
        s += wo[64 + j] * l2x[j]
    raw = np.int64(s) + LS_OUT_BQ[bucket] + skip

    wpsqt = np.float64(white_psqt[bucket])
    bpsqt = np.float64(black_psqt[bucket])
    # 600/16384 = 75/2048 and 600/9600 = 1/16 are dyadic, so on these integers the float64 sum
    # is exact and the truncation is unambiguous on any IEEE machine
    val = np.float64(raw) * (600.0 / 16384.0) + (wpsqt - bpsqt) * (us - 0.5) * (600.0 / 9600.0)
    return np.int64(math.trunc(val))
