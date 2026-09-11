"""Bullet-trained Chess768 network: 768 inputs (12 piece kinds x 64 squares, no king buckets),
1024 hidden per perspective, SCReLU, 8 output buckets by piece count. Same interface as the
HalfKAv2_hm module (nnue_init_root, nnue_make_move, nnue_make_null, nnue_eval,
new_accumulator_stack). Layout: friendly pieces first (0-5) then enemy (6-11), squares flipped
vertically for the black perspective, l1w decoded as (2048, 8) transposed, side-to-move half
first. nnue_eval returns int64 centipawns using the network's training-time scale.
"""
import math
import os

import numpy as np
from numba import njit

from board import bit, lsb

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")
NNUE_AVAILABLE = os.path.exists(_WEIGHTS_PATH)
NNUE_STACK_SIZE = 128
L0_HIDDEN = 1024
NUM_BUCKETS = 8
SCORE_PER_UNIT = 400.0

if NNUE_AVAILABLE:
    _npz = np.load(_WEIGHTS_PATH)
    FT_WEIGHT = np.ascontiguousarray(_npz["l0w"], dtype=np.int16)                                     # (768, 1024), x255
    FT_BIAS = np.ascontiguousarray(_npz["l0b"], dtype=np.int16)                                       # (1024,), x255
    # an accumulator is the bias plus at most 32 rows (33 transiently inside a diff); numba integer
    # overflow is silent, so check that even the 33 largest weights of every column fit int16
    _bound = np.partition(np.abs(FT_WEIGHT), FT_WEIGHT.shape[0] - 33, axis=0)[-33:].astype(np.int32).sum(axis=0)
    if int((_bound + np.abs(FT_BIAS).astype(np.int32)).max()) >= 32767:
        raise RuntimeError("int16 accumulator could overflow")
    del _bound
    L1_WEIGHT = np.ascontiguousarray(_npz["l1w"].astype(np.float32).ravel().reshape(2048, 8).T)        # (8, 2048)
    L1_BIAS = np.ascontiguousarray(_npz["l1b"].astype(np.float32))                                    # (8,)
else:
    FT_WEIGHT = np.zeros((768, 1024), dtype=np.int16)
    FT_BIAS = np.zeros(1024, dtype=np.int16)
    L1_WEIGHT = np.zeros((8, 2048), dtype=np.float32)
    L1_BIAS = np.zeros(8, dtype=np.float32)

_F0 = np.float32(0.0)
_F1 = np.float32(1.0)
_INV255 = np.float32(1.0 / 255.0)  # the int16 accumulator carries the activation x255


@njit(cache=False, fastmath=True)
def _feature_index(is_white_pov, sq, piece_type, piece_is_white):  # type: ignore[no-untyped-def]
    friendly = piece_is_white == is_white_pov
    base = piece_type - 1 if friendly else piece_type + 5
    return 64 * base + (sq if is_white_pov else (sq ^ 56))


@njit(cache=False, fastmath=True)
def refresh_accumulator(bb, is_white_pov, acc_out):  # type: ignore[no-untyped-def]
    acc_out[:] = FT_BIAS
    for colour in range(2):
        piece_is_white = colour == 0
        for piece_type in range(1, 7):
            pieces = bb[piece_type + 1] & bb[colour]
            while pieces:
                sq = lsb(pieces)
                pieces &= pieces - 1
                acc_out += FT_WEIGHT[_feature_index(is_white_pov, sq, piece_type, piece_is_white)]


@njit(cache=False, fastmath=True)
def nnue_init_root(bb, acc, psqt):  # type: ignore[no-untyped-def]
    refresh_accumulator(bb, True, acc[0, 0])
    refresh_accumulator(bb, False, acc[0, 1])


@njit(cache=False, fastmath=True)
def _apply_diff(bbs, mbs, ply, is_white_pov, acc_out):  # type: ignore[no-untyped-def]
    """Update one perspective's accumulator (a 1-D view, so the row adds vectorise) from the
    squares whose contents changed between ply and ply+1 (XOR of the 8 bitboard planes)."""
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
            acc_out -= FT_WEIGHT[_feature_index(is_white_pov, sq, before_piece, before_white)]
        if after_piece != 0:
            acc_out += FT_WEIGHT[_feature_index(is_white_pov, sq, after_piece, after_white)]


@njit(cache=False, fastmath=True)
def _fused_update(bbs, mbs, ply, is_white_pov, old, new):  # type: ignore[no-untyped-def]
    """new = old - removed rows + added rows in one pass over the accumulator. Integer
    arithmetic, so the result is bit-identical to copying and then subtracting and adding row
    by row. A move removes at most two features and adds at most two (castling, capture,
    en passant, promotion); anything else falls back to the row-by-row path."""
    changed = np.int64(0)
    for plane in range(8):
        changed |= bbs[ply, plane] ^ bbs[ply + 1, plane]
    ns = 0; na = 0; s0 = 0; s1 = 0; a0 = 0; a1 = 0
    while changed:
        sq = lsb(changed)
        changed &= changed - 1
        before_piece = mbs[ply, sq]
        after_piece = mbs[ply + 1, sq]
        if before_piece != 0:
            f = _feature_index(is_white_pov, sq, before_piece, (bbs[ply, 0] & bit(sq)) != 0)
            if ns == 0:
                s0 = f
            else:
                s1 = f
            ns += 1
        if after_piece != 0:
            f = _feature_index(is_white_pov, sq, after_piece, (bbs[ply + 1, 0] & bit(sq)) != 0)
            if na == 0:
                a0 = f
            else:
                a1 = f
            na += 1
    if ns == 1 and na == 1:
        w0 = FT_WEIGHT[s0]; v0 = FT_WEIGHT[a0]
        for i in range(L0_HIDDEN):
            new[i] = old[i] - w0[i] + v0[i]
    elif ns == 2 and na == 1:
        w0 = FT_WEIGHT[s0]; w1 = FT_WEIGHT[s1]; v0 = FT_WEIGHT[a0]
        for i in range(L0_HIDDEN):
            new[i] = old[i] - w0[i] - w1[i] + v0[i]
    elif ns == 2 and na == 2:
        w0 = FT_WEIGHT[s0]; w1 = FT_WEIGHT[s1]; v0 = FT_WEIGHT[a0]; v1 = FT_WEIGHT[a1]
        for i in range(L0_HIDDEN):
            new[i] = old[i] - w0[i] - w1[i] + v0[i] + v1[i]
    else:
        new[:] = old
        _apply_diff(bbs, mbs, ply, is_white_pov, new)


@njit(cache=False, fastmath=True)
def nnue_make_move(bbs, mbs, acc, psqt, ply):  # type: ignore[no-untyped-def]
    """Incremental update; no king-move refresh is needed since features do not depend on the
    king square."""
    _fused_update(bbs, mbs, ply, True, acc[ply, 0], acc[ply + 1, 0])
    _fused_update(bbs, mbs, ply, False, acc[ply, 1], acc[ply + 1, 1])


@njit(cache=False, fastmath=True)
def nnue_make_null(acc, psqt, ply):  # type: ignore[no-untyped-def]
    acc[ply + 1, 0] = acc[ply, 0]
    acc[ply + 1, 1] = acc[ply, 1]


@njit(cache=False, fastmath=True)
def forward(white_acc, black_acc, side, piece_count):  # type: ignore[no-untyped-def]
    if side == 0:
        stm = white_acc; ntm = black_acc
    else:
        stm = black_acc; ntm = white_acc
    bucket = (piece_count - 2) // 4
    if bucket < 0:
        bucket = 0
    if bucket > NUM_BUCKETS - 1:
        bucket = NUM_BUCKETS - 1
    w = L1_WEIGHT[bucket]
    s = _F0
    for i in range(L0_HIDDEN):
        a = min(max(np.float32(stm[i]) * _INV255, _F0), _F1)
        s += w[i] * (a * a)
    for i in range(L0_HIDDEN):
        b = min(max(np.float32(ntm[i]) * _INV255, _F0), _F1)
        s += w[L0_HIDDEN + i] * (b * b)
    return np.int64(math.trunc(np.float64(s + L1_BIAS[bucket]) * SCORE_PER_UNIT))


@njit(cache=False, fastmath=True)
def nnue_eval(acc, psqt, side, piece_count, ply):  # type: ignore[no-untyped-def]
    return forward(acc[ply, 0], acc[ply, 1], side, piece_count)


def new_accumulator_stack():
    """psqt is a dummy kept for interface compatibility; this network has no PSQT outputs."""
    acc = np.zeros((NNUE_STACK_SIZE, 2, L0_HIDDEN), dtype=np.int16)
    psqt = np.zeros((NNUE_STACK_SIZE, 2, 1), dtype=np.int32)   # dummy, matches the search's signature
    return acc, psqt
