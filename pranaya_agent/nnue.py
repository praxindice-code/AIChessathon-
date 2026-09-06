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

MAX_FT_ACTIVATION = np.float32(255.0 / 256.0)
MAX_HIDDEN_ACTIVATION = np.float32(127.0 / 128.0)
L0_CORRECTION_FACTOR = np.float32(1.0)
SQR_CRELU_CORRECTION_FACTOR = np.float32(1.0)

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")
NNUE_AVAILABLE = os.path.exists(_WEIGHTS_PATH)

if NNUE_AVAILABLE:
    _npz = np.load(_WEIGHTS_PATH)
    FT_WEIGHT = _npz["ft_weight"].astype(np.float32)
    FT_PSQT_WEIGHT = _npz["ft_psqt_weight"].astype(np.float32)
    FT_BIAS = _npz["ft_bias"].astype(np.float32)
    LS_L1_WEIGHT = _npz["ls_l1_weight"].astype(np.float32)
    LS_L1_BIAS = _npz["ls_l1_bias"].astype(np.float32)
    LS_L2_WEIGHT = _npz["ls_l2_weight"].astype(np.float32)
    LS_L2_BIAS = _npz["ls_l2_bias"].astype(np.float32)
    LS_OUT_WEIGHT = _npz["ls_out_weight"].astype(np.float32)
    LS_OUT_BIAS = _npz["ls_out_bias"].astype(np.float32)
    L1 = int(_npz["meta_L1"])
    NUM_PSQT_BUCKETS = int(_npz["meta_num_psqt_buckets"])
else:
    # placeholder zero-size arrays so the module still imports cleanly and njit functions below
    # still compile (numba needs concrete array types even if they're never actually reached)
    FT_WEIGHT = np.zeros((1, 1024), dtype=np.float32)
    FT_PSQT_WEIGHT = np.zeros((1, 8), dtype=np.float32)
    FT_BIAS = np.zeros(1024, dtype=np.float32)
    LS_L1_WEIGHT = np.zeros((8, 32, 1024), dtype=np.float32)
    LS_L1_BIAS = np.zeros((8, 32), dtype=np.float32)
    LS_L2_WEIGHT = np.zeros((8, 32, 64), dtype=np.float32)
    LS_L2_BIAS = np.zeros((8, 32), dtype=np.float32)
    LS_OUT_WEIGHT = np.zeros((8, 1, 128), dtype=np.float32)
    LS_OUT_BIAS = np.zeros((8, 1), dtype=np.float32)
    L1 = 1024
    NUM_PSQT_BUCKETS = 8


NNUE_STACK_SIZE = 128  # safe upper bound over search.py's actual STACK (104); avoids a
                        # circular import just to match it exactly


@njit(cache=False)
def nnue_init_root(bb, acc, psqt):  # type: ignore[no-untyped-def]
    """Call once, at the very start of a search, to populate ply 0's accumulators from
    scratch. `acc` is shaped (STACK, 2, L1), `psqt` (STACK, 2, num_psqt_buckets) -- both
    allocated by the caller (agent.py) and passed in explicitly, the same way bbs/sts/mbs
    already are, since numba freezes module-level array globals as read-only inside njit
    functions and there is no way to mutate one from here otherwise."""
    refresh_accumulator(bb, True, acc[0, 0], psqt[0, 0])
    refresh_accumulator(bb, False, acc[0, 1], psqt[0, 1])


@njit(cache=False)
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


@njit(cache=False)
def _apply_diff(bbs, mbs, ply, is_white_pov, acc_out, psqt_out):  # type: ignore[no-untyped-def]
    pov_colour = 0 if is_white_pov else 1
    king_sq = lsb(bbs[ply, 7] & bbs[ply, pov_colour])
    for sq in range(64):
        before_piece = mbs[ply, sq]
        after_piece = mbs[ply + 1, sq]
        before_white = (bbs[ply, 0] & bit(sq)) != 0
        after_white = (bbs[ply + 1, 0] & bit(sq)) != 0
        # mbs alone isn't enough to detect "nothing changed here": a same-type capture (pawn
        # takes pawn, most commonly) leaves the piece TYPE identical while the colour flips, so
        # the identity check has to include colour too, or a capture like that gets silently
        # treated as a no-op and the accumulator never gets updated for that square at all
        if before_piece == after_piece and (before_piece == 0 or before_white == after_white):
            continue
        if before_piece != 0:
            idx = perspective_feature(is_white_pov, sq, before_piece, before_white, king_sq)
            acc_out -= FT_WEIGHT[idx]
            psqt_out -= FT_PSQT_WEIGHT[idx]
        if after_piece != 0:
            idx = perspective_feature(is_white_pov, sq, after_piece, after_white, king_sq)
            acc_out += FT_WEIGHT[idx]
            psqt_out += FT_PSQT_WEIGHT[idx]


@njit(cache=False)
def nnue_make_null(acc, psqt, ply):  # type: ignore[no-untyped-def]
    """Null move: no piece placement changes, so both perspectives' accumulators carry over
    unchanged."""
    acc[ply + 1, 0] = acc[ply, 0]
    acc[ply + 1, 1] = acc[ply, 1]
    psqt[ply + 1, 0] = psqt[ply, 0]
    psqt[ply + 1, 1] = psqt[ply, 1]


@njit(cache=False)
def nnue_eval(acc, psqt, side, piece_count, ply):  # type: ignore[no-untyped-def]
    """Reads this ply's already-maintained accumulators (via nnue_make_move/nnue_make_null)
    and runs the forward pass. Does not touch the board at all -- purely a function of
    whatever's already sitting in acc/psqt[ply]."""
    return forward(acc[ply, 0], acc[ply, 1], psqt[ply, 0], psqt[ply, 1], side, piece_count)


def new_accumulator_stack():
    """Allocates a fresh (acc, psqt) pair sized for one full search stack. Call once per
    independent search context (agent.py needs a separate pair for the main search and for the
    pondering thread, exactly like PONDER_BBS is separate from BBS)."""
    if NNUE_AVAILABLE:
        acc = np.zeros((NNUE_STACK_SIZE, 2, L1), dtype=np.float32)
        psqt = np.zeros((NNUE_STACK_SIZE, 2, NUM_PSQT_BUCKETS), dtype=np.float32)
    else:
        acc = np.zeros((NNUE_STACK_SIZE, 2, 1), dtype=np.float32)
        psqt = np.zeros((NNUE_STACK_SIZE, 2, 1), dtype=np.float32)
    return acc, psqt


@njit(cache=False)
def orient(is_white_pov, sq, ksq):  # type: ignore[no-untyped-def]
    kfile = ksq & 7
    flip_h = 7 if kfile < 4 else 0
    flip_v = 0 if is_white_pov else 56
    return (flip_h ^ flip_v) ^ sq


@njit(cache=False)
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


@njit(cache=False)
def refresh_accumulator(bb, is_white_pov, acc_out, psqt_out):  # type: ignore[no-untyped-def]
    """Full recompute of one perspective's raw accumulator + psqt vector from a bb array."""
    pov_colour = 0 if is_white_pov else 1
    king_sq = lsb(bb[7] & bb[pov_colour])

    acc_out[:] = FT_BIAS
    psqt_out[:] = 0.0

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


@njit(cache=False)
def forward(white_acc, black_acc, white_psqt, black_psqt, side, piece_count):  # type: ignore[no-untyped-def]
    """Combines both perspectives' raw accumulators into a final centipawn-ish score. `side`:
    0 = white to move, 1 = black to move."""
    us = np.float32(1.0) if side == 0 else np.float32(0.0)

    if side == 0:
        own_raw, opp_raw = white_acc, black_acc
    else:
        own_raw, opp_raw = black_acc, white_acc

    l0 = np.empty(2 * L1, dtype=np.float32)
    for i in range(L1):
        v = own_raw[i]
        l0[i] = min(max(v, np.float32(0.0)), MAX_FT_ACTIVATION)
    for i in range(L1):
        v = opp_raw[i]
        l0[L1 + i] = min(max(v, np.float32(0.0)), MAX_FT_ACTIVATION)

    q = L1 // 2
    l0_final = np.empty(L1, dtype=np.float32)
    for i in range(q):
        l0_final[i] = l0[i] * l0[q + i]
    for i in range(q):
        l0_final[q + i] = l0[L1 + i] * l0[L1 + q + i]
    for i in range(L1):
        l0_final[i] *= L0_CORRECTION_FACTOR

    bucket = (piece_count - 1) // 4
    if bucket >= NUM_PSQT_BUCKETS:
        bucket = NUM_PSQT_BUCKETS - 1  # defensive clamp; shouldn't trigger in real chess

    # Explicit multiply-accumulate loops replace `@` matmul so numba can compile without
    # BLAS (numba's `@` implementation calls into scipy.linalg's BLAS bindings, and the
    # chessathon sandbox's Python doesn't ship scipy). The math is identical.
    l1c = np.empty(32, dtype=np.float32)
    for i in range(32):
        s = np.float32(0.0)
        for j in range(L1):
            s += LS_L1_WEIGHT[bucket, i, j] * l0_final[j]
        l1c[i] = s + LS_L1_BIAS[bucket, i]
    skip = l1c[-2] - l1c[-1]

    l1x = np.empty(64, dtype=np.float32)
    for i in range(32):
        sq = l1c[i] * l1c[i] * SQR_CRELU_CORRECTION_FACTOR
        l1x[i] = min(max(sq, np.float32(0.0)), MAX_HIDDEN_ACTIVATION)
    for i in range(32):
        l1x[32 + i] = min(max(l1c[i], np.float32(0.0)), MAX_HIDDEN_ACTIVATION)

    l2c = np.empty(32, dtype=np.float32)
    for i in range(32):
        s = np.float32(0.0)
        for j in range(64):
            s += LS_L2_WEIGHT[bucket, i, j] * l1x[j]
        l2c[i] = s + LS_L2_BIAS[bucket, i]

    l2x = np.empty(64, dtype=np.float32)
    for i in range(32):
        sq = l2c[i] * l2c[i] * SQR_CRELU_CORRECTION_FACTOR
        l2x[i] = min(max(sq, np.float32(0.0)), MAX_HIDDEN_ACTIVATION)
    for i in range(32):
        l2x[32 + i] = min(max(l2c[i], np.float32(0.0)), MAX_HIDDEN_ACTIVATION)

    l3_input = np.empty(128, dtype=np.float32)
    l3_input[:64] = l1x
    l3_input[64:] = l2x

    l3c = np.float32(0.0)
    for j in range(128):
        l3c += LS_OUT_WEIGHT[bucket, 0, j] * l3_input[j]
    l3c += LS_OUT_BIAS[bucket, 0]
    raw_output = l3c + skip

    wpsqt = white_psqt[bucket]
    bpsqt = black_psqt[bucket]
    return raw_output + (wpsqt - bpsqt) * (us - np.float32(0.5))
