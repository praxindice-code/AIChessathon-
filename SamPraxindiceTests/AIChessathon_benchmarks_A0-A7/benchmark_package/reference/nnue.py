"""Numba port of this project's HalfKAv2_hm^ network, matching search.py's existing NNUE
interface exactly (nnue_init_root, nnue_make_move, nnue_make_null, nnue_eval,
new_accumulator_stack), built from and validated against a pure-Python reference
implementation that was itself checked to full precision against the real trained model's own
forward() on multiple positions (see halfka_reference.py, ground_truth_check.py,
debug_l0.py from the validation session).

Unlike Bullet's Chess768 (no king dependency at all), HalfKAv2_hm's feature indices are
king-bucket-dependent: EVERY feature index for a perspective depends on that perspective's own
king square. This means a king move requires a FULL accumulator refresh for that perspective
(every input feature changes), not an incremental diff -- the same requirement the original
HalfKAv2_hm nnue.py in this project had. Non-king moves use the fast incremental diff path,
per perspective independently (only the moved perspective needs updating if its own king
didn't move; the other perspective's accumulator is unaffected by a move that doesn't touch
squares relevant to ITS orientation... actually both perspectives' accumulators generally need
updating for any move that changes the board, since both track the same underlying piece
placement, just oriented differently. Only king moves specifically require a FULL refresh
rather than an incremental diff, because the king SQUARE determines the bucket for ALL 768
planes of that perspective, not just the squares that actually changed.)
"""

import os

import numpy as np
from numba import njit

from board import bit, lsb, popcount

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halfka_raw.npz")
_QUANTIZED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halfka_quantized_raw_8bit.npz")
NNUE_AVAILABLE = os.path.exists(_WEIGHTS_PATH) or os.path.exists(_QUANTIZED_PATH)

NNUE_STACK_SIZE = 128
L1_SIZE = 1024  # per-perspective hidden width (excluding PSQT)
NUM_PSQT = 8
ACC_WIDTH = L1_SIZE + NUM_PSQT  # 1032
NUM_LS_BUCKETS = 8
L2_WIDTH = 32
L3_INPUT_WIDTH = 128  # L2*2 + L3*2 = 32*2 + 32*2

MAX_FT_ACTIVATION = np.float32(255.0 / 256.0)
MAX_HIDDEN_ACTIVATION = np.float32(127.0 / 128.0)

# fmt: off
_KING_BUCKETS_PY = [
  -1, -1, -1, -1, 31, 30, 29, 28,
  -1, -1, -1, -1, 27, 26, 25, 24,
  -1, -1, -1, -1, 23, 22, 21, 20,
  -1, -1, -1, -1, 19, 18, 17, 16,
  -1, -1, -1, -1, 15, 14, 13, 12,
  -1, -1, -1, -1, 11, 10, 9, 8,
  -1, -1, -1, -1, 7, 6, 5, 4,
  -1, -1, -1, -1, 3, 2, 1, 0
]
# fmt: on
KING_BUCKETS = np.array(_KING_BUCKETS_PY, dtype=np.int64)

if NNUE_AVAILABLE:
    if os.path.exists(_WEIGHTS_PATH):
        _npz = np.load(_WEIGHTS_PATH)
        FT_WEIGHT = _npz["ft_weight"].astype(np.float32)
        FT_BIAS = _npz["ft_bias"].astype(np.float32)
        L1_WEIGHT = _npz["l1w"].astype(np.float32)
        L1_BIAS = _npz["l1b"].astype(np.float32)
        L2_WEIGHT = _npz["l2w"].astype(np.float32)
        L2_BIAS = _npz["l2b"].astype(np.float32)
        OUT_WEIGHT = _npz["outw"].astype(np.float32)
        OUT_BIAS = _npz["outb"].astype(np.float32)
        NNUE2SCORE = float(_npz["nnue2score"][0])
    else:
        # int8-for-storage-only: dequantize to float32 here, once, at import. All downstream
        # computation is then byte-for-byte the same code path as the full-precision case --
        # validated separately in test_int8_accuracy.py, which compares this dequantized
        # result directly against the full-precision reference on real positions.
        _npz = np.load(_QUANTIZED_PATH)

        def _dq(key):  # type: ignore[no-untyped-def]
            return _npz[key].astype(np.float32) / _npz[f"{key}_scale"]

        FT_WEIGHT = _dq("ft_weight")
        FT_BIAS = _dq("ft_bias")
        L1_WEIGHT = _dq("l1w")
        L1_BIAS = _dq("l1b")
        L2_WEIGHT = _dq("l2w")
        L2_BIAS = _dq("l2b")
        OUT_WEIGHT = _dq("outw")
        OUT_BIAS = _dq("outb")
        NNUE2SCORE = float(_npz["nnue2score"][0])
else:
    FT_WEIGHT = np.zeros((1, 1), dtype=np.float32)
    FT_BIAS = np.zeros(1, dtype=np.float32)
    L1_WEIGHT = np.zeros((1, 1), dtype=np.float32)
    L1_BIAS = np.zeros(1, dtype=np.float32)
    L2_WEIGHT = np.zeros((1, 1), dtype=np.float32)
    L2_BIAS = np.zeros(1, dtype=np.float32)
    OUT_WEIGHT = np.zeros((1, 1), dtype=np.float32)
    OUT_BIAS = np.zeros(1, dtype=np.float32)
    NNUE2SCORE = 600.0


@njit(cache=False, fastmath=True, inline="always")
def _orient(is_white_pov, sq, ksq):  # type: ignore[no-untyped-def]
    kfile = ksq % 8
    h = 7 if kfile < 4 else 0
    v = 56 if not is_white_pov else 0
    return h ^ v ^ sq


@njit(cache=False, fastmath=True, inline="always")
def _feature_index(is_white_pov, king_sq, sq, piece_type, piece_is_white):  # type: ignore[no-untyped-def]
    """Directly matches halfka_v2_hm.py's _halfka_idx: friendly/enemy INTERLEAVED per piece
    type (pawn-friendly=0, pawn-enemy=1, knight-friendly=2, ...), not block-separated."""
    friendly = piece_is_white == is_white_pov
    p_idx = (piece_type - 1) * 2 + (0 if friendly else 1)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KING_BUCKETS[o_ksq] * 768


@njit(cache=False, fastmath=True)
def _piece_type_at(bb, sq):  # type: ignore[no-untyped-def]
    b = bit(sq)
    for pt in range(1, 7):
        if bb[pt + 1] & b:
            return pt
    return 0


@njit(cache=False, fastmath=True)
def refresh_accumulator(bb, is_white_pov, acc_out):  # type: ignore[no-untyped-def]
    """Full recompute from scratch -- used at root init and whenever this perspective's own
    king moves, since the king square determines the bucket for every one of the 768 planes."""
    acc_out[:] = FT_BIAS
    king_colour = 0 if is_white_pov else 1
    king_sq = lsb(bb[7] & bb[king_colour])
    occ = bb[0] | bb[1]
    while occ:
        sq = lsb(occ)
        occ &= occ - 1
        piece_type = _piece_type_at(bb, sq)
        piece_is_white = (bb[0] & bit(sq)) != 0
        idx = _feature_index(is_white_pov, king_sq, sq, piece_type, piece_is_white)
        acc_out += FT_WEIGHT[idx]


@njit(cache=False, fastmath=True)
def nnue_init_root(bb, acc, psqt):  # type: ignore[no-untyped-def]
    refresh_accumulator(bb, True, acc[0, 0])
    refresh_accumulator(bb, False, acc[0, 1])


@njit(cache=False, fastmath=True)
def nnue_make_move(bbs, mbs, acc, psqt, ply):  # type: ignore[no-untyped-def]
    white_king_before = lsb(bbs[ply, 7] & bbs[ply, 0])
    white_king_after = lsb(bbs[ply + 1, 7] & bbs[ply + 1, 0])
    black_king_before = lsb(bbs[ply, 7] & bbs[ply, 1])
    black_king_after = lsb(bbs[ply + 1, 7] & bbs[ply + 1, 1])

    white_king_moved = white_king_before != white_king_after
    black_king_moved = black_king_before != black_king_after

    if white_king_moved:
        refresh_accumulator(bbs[ply + 1], True, acc[ply + 1, 0])
    else:
        acc[ply + 1, 0] = acc[ply, 0]

    if black_king_moved:
        refresh_accumulator(bbs[ply + 1], False, acc[ply + 1, 1])
    else:
        acc[ply + 1, 1] = acc[ply, 1]

    if white_king_moved and black_king_moved:
        return  # both perspectives fully refreshed, nothing left to do

    changed = np.int64(0)
    for plane in range(8):
        changed |= bbs[ply, plane] ^ bbs[ply + 1, plane]

    white_acc = acc[ply + 1, 0]
    black_acc = acc[ply + 1, 1]
    white_king_sq = white_king_after
    black_king_sq = black_king_after

    while changed:
        sq = lsb(changed)
        changed &= changed - 1

        before_piece = mbs[ply, sq]
        after_piece = mbs[ply + 1, sq]
        before_white = (bbs[ply, 0] & bit(sq)) != 0
        after_white = (bbs[ply + 1, 0] & bit(sq)) != 0
        if before_piece == after_piece and (before_piece == 0 or before_white == after_white):
            continue

        if before_piece != 0:
            if not white_king_moved:
                w_idx = _feature_index(True, white_king_sq, sq, before_piece, before_white)
                white_acc -= FT_WEIGHT[w_idx]
            if not black_king_moved:
                b_idx = _feature_index(False, black_king_sq, sq, before_piece, before_white)
                black_acc -= FT_WEIGHT[b_idx]
        if after_piece != 0:
            if not white_king_moved:
                w_idx = _feature_index(True, white_king_sq, sq, after_piece, after_white)
                white_acc += FT_WEIGHT[w_idx]
            if not black_king_moved:
                b_idx = _feature_index(False, black_king_sq, sq, after_piece, after_white)
                black_acc += FT_WEIGHT[b_idx]


@njit(cache=False, fastmath=True)
def nnue_make_null(acc, psqt, ply):  # type: ignore[no-untyped-def]
    acc[ply + 1, 0] = acc[ply, 0]
    acc[ply + 1, 1] = acc[ply, 1]


@njit(cache=False, fastmath=True)
def forward(white_acc, black_acc, side, piece_count):  # type: ignore[no-untyped-def]
    """side: 0 for white to move, 1 for black. piece_count: total pieces on board."""
    us = np.float32(1.0) if side == 0 else np.float32(0.0)
    them = np.float32(1.0) - us

    w = white_acc[:L1_SIZE]
    wpsqt = white_acc[L1_SIZE:]
    b = black_acc[:L1_SIZE]
    bpsqt = black_acc[L1_SIZE:]

    # l0_pre = us*cat([w,b]) + them*cat([b,w]) -- matches double_ft_functions.py exactly
    l0_pre = np.empty(2 * L1_SIZE, dtype=np.float32)
    for i in range(L1_SIZE):
        l0_pre[i] = us * w[i] + them * b[i]
        l0_pre[L1_SIZE + i] = us * b[i] + them * w[i]

    for i in range(2 * L1_SIZE):
        v = l0_pre[i]
        if v < 0.0:
            v = 0.0
        elif v > MAX_FT_ACTIVATION:
            v = MAX_FT_ACTIVATION
        l0_pre[i] = v

    l0 = np.empty(L1_SIZE, dtype=np.float32)
    quarter = L1_SIZE // 2  # 512
    for i in range(quarter):
        l0[i] = l0_pre[i] * l0_pre[quarter + i]
        l0[quarter + i] = l0_pre[2 * quarter + i] * l0_pre[3 * quarter + i]

    bucket_idx = (piece_count - 1) // 4
    if bucket_idx < 0:
        bucket_idx = 0
    if bucket_idx > NUM_PSQT - 1:
        bucket_idx = NUM_PSQT - 1

    ls_bucket = bucket_idx  # same bucket formula for both, per model.py's calculate_buckets

    l1c = np.empty(L2_WIDTH, dtype=np.float32)
    l1_row_offset = ls_bucket * L2_WIDTH
    for o in range(L2_WIDTH):
        acc_val = L1_BIAS[l1_row_offset + o]
        row = L1_WEIGHT[l1_row_offset + o]
        for i in range(L1_SIZE):
            acc_val += row[i] * l0[i]
        l1c[o] = acc_val

    l1x_out = l1c[L2_WIDTH - 2] - l1c[L2_WIDTH - 1]

    l1x = np.empty(2 * L2_WIDTH, dtype=np.float32)  # 64
    for i in range(L2_WIDTH):
        sq = l1c[i] * l1c[i]
        if sq > MAX_HIDDEN_ACTIVATION:
            sq = MAX_HIDDEN_ACTIVATION
        l1x[i] = sq
        v = l1c[i]
        if v < 0.0:
            v = 0.0
        elif v > MAX_HIDDEN_ACTIVATION:
            v = MAX_HIDDEN_ACTIVATION
        l1x[L2_WIDTH + i] = v

    l2c = np.empty(L2_WIDTH, dtype=np.float32)
    l2_row_offset = ls_bucket * L2_WIDTH
    for o in range(L2_WIDTH):
        acc_val = L2_BIAS[l2_row_offset + o]
        row = L2_WEIGHT[l2_row_offset + o]
        for i in range(2 * L2_WIDTH):
            acc_val += row[i] * l1x[i]
        l2c[o] = acc_val

    l2x = np.empty(2 * L2_WIDTH, dtype=np.float32)  # 64
    for i in range(L2_WIDTH):
        sq = l2c[i] * l2c[i]
        if sq > MAX_HIDDEN_ACTIVATION:
            sq = MAX_HIDDEN_ACTIVATION
        l2x[i] = sq
        v = l2c[i]
        if v < 0.0:
            v = 0.0
        elif v > MAX_HIDDEN_ACTIVATION:
            v = MAX_HIDDEN_ACTIVATION
        l2x[L2_WIDTH + i] = v

    l3_input = np.empty(L3_INPUT_WIDTH, dtype=np.float32)  # 128
    for i in range(2 * L2_WIDTH):
        l3_input[i] = l1x[i]
        l3_input[2 * L2_WIDTH + i] = l2x[i]

    out_row = OUT_WEIGHT[ls_bucket]
    l3c = OUT_BIAS[ls_bucket]
    for i in range(L3_INPUT_WIDTH):
        l3c += out_row[i] * l3_input[i]

    l3x = l3c + l1x_out
    final = l3x + (wpsqt[bucket_idx] - bpsqt[bucket_idx]) * (us - np.float32(0.5))

    return final


@njit(cache=False, fastmath=True)
def nnue_eval(acc, psqt, side, piece_count, ply):  # type: ignore[no-untyped-def]
    raw = forward(acc[ply, 0], acc[ply, 1], side, piece_count)
    return np.int64(raw * NNUE2SCORE)


def new_accumulator_stack():
    """psqt is a tiny unused dummy here, kept only so the (acc, psqt) tuple shape matches
    what agent.py already expects to unpack and pass around -- PSQT is actually stored inside
    the last 8 elements of each 1032-wide accumulator row, not in this separate array."""
    if NNUE_AVAILABLE:
        acc = np.zeros((NNUE_STACK_SIZE, 2, ACC_WIDTH), dtype=np.float32)
        psqt = np.zeros((NNUE_STACK_SIZE, 2, 1), dtype=np.float32)
    else:
        acc = np.zeros((1, 2, 1), dtype=np.float32)
        psqt = np.zeros((1, 2, 1), dtype=np.float32)
    return acc, psqt
