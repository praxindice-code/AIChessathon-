"""Tapered evaluation: material, piece-square tables, mobility, pawn structure, king safety.

Everything is scored from White's point of view in two phases, a middlegame score and an
endgame score, and interpolated on the material left on the board. The caller gets the value
from the side to move's point of view, which is what negamax wants.

The piece-square tables below are written as a picture of the board from White's side, a8
first, so they can be read and edited as a board. _flip turns them into a1-first array.
"""

import numpy as np
from numba import njit

from board import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    NOT_FILE_A,
    NOT_FILE_H,
    bishop_attacks,
    i64,
    king_square,
    lsb,
    popcount,
    queen_attacks,
    rook_attacks,
    srl,
)

MATE = 30000
MATE_IN_MAX = MATE - 1000
DRAW = 0

# midgame and endgame material, index by piece type
MATERIAL_MG = np.array([0, 100, 320, 335, 500, 975, 0], dtype=np.int32)
MATERIAL_EG = np.array([0, 118, 318, 342, 545, 1010, 0], dtype=np.int32)
# knight 1, bishop 1, rook 2, queen 4, summing to 24 at the full opening array
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
TOTAL_PHASE = 24


def _flip(table: list[int]) -> list[int]:
    """a8-first picture order into a1-first square order."""
    out = [0] * 64
    for row in range(8):
        for col in range(8):
            out[(7 - row) * 8 + col] = table[row * 8 + col]
    return out


# fmt: off
PAWN_MG = _flip([
      0,   0,   0,   0,   0,   0,   0,   0,
     90,  95,  95,  95,  95,  95,  95,  90,
     35,  40,  50,  60,  60,  50,  40,  35,
     12,  16,  24,  38,  38,  22,  16,  12,
      4,   8,  14,  28,  28,  10,   8,   4,
      2,   4,   4,   6,   6,  -2,   4,   2,
      4,   8,   8, -18, -18,  10,  10,   4,
      0,   0,   0,   0,   0,   0,   0,   0,
])
PAWN_EG = _flip([
      0,   0,   0,   0,   0,   0,   0,   0,
    150, 145, 140, 130, 130, 140, 145, 150,
     85,  82,  74,  62,  62,  70,  80,  85,
     42,  38,  32,  26,  26,  30,  36,  40,
     18,  16,  10,   8,   8,  10,  14,  16,
      6,   6,   2,   2,   2,   2,   4,   4,
      8,   6,   6,   8,   8,   4,   4,   4,
      0,   0,   0,   0,   0,   0,   0,   0,
])
KNIGHT_MG = _flip([
    -60, -40, -25, -20, -20, -25, -40, -60,
    -35, -15,   8,  14,  14,   8, -15, -35,
    -20,  10,  26,  32,  32,  26,  10, -20,
    -16,  10,  28,  36,  36,  28,  10, -16,
    -18,   6,  24,  30,  30,  24,   6, -18,
    -24,   4,  16,  20,  20,  18,   4, -24,
    -38, -18,   2,   8,   8,   2, -18, -38,
    -65, -32, -22, -16, -16, -22, -32, -65,
])
KNIGHT_EG = _flip([
    -50, -32, -16, -12, -12, -16, -32, -50,
    -28, -12,   0,   6,   6,   0, -12, -28,
    -16,   0,  12,  18,  18,  12,   0, -16,
    -12,   6,  18,  24,  24,  18,   6, -12,
    -12,   6,  18,  24,  24,  18,   6, -12,
    -16,   0,  12,  16,  16,  12,   0, -16,
    -28, -14,   0,   6,   6,   0, -14, -28,
    -48, -30, -18, -12, -12, -18, -30, -48,
])
BISHOP_MG = _flip([
    -20, -12,  -8,  -8,  -8,  -8, -12, -20,
    -10,   4,   2,   0,   0,   2,   4, -10,
     -6,   8,  12,  12,  12,  12,   8,  -6,
     -4,   6,  14,  20,  20,  14,   6,  -4,
     -4,   8,  14,  20,  20,  14,   8,  -4,
     -6,  14,  14,  14,  14,  14,  14,  -6,
     -8,  16,   6,   6,   6,   6,  16,  -8,
    -22,  -8, -14, -14, -14, -14,  -8, -22,
])
BISHOP_EG = _flip([
    -16,  -8,  -6,  -4,  -4,  -6,  -8, -16,
     -8,   0,   2,   4,   4,   2,   0,  -8,
     -4,   4,   8,  10,  10,   8,   4,  -4,
     -2,   6,  10,  14,  14,  10,   6,  -2,
     -2,   6,  10,  14,  14,  10,   6,  -2,
     -4,   4,   8,  10,  10,   8,   4,  -4,
     -8,   0,   2,   4,   4,   2,   0,  -8,
    -16,  -8,  -6,  -4,  -4,  -6,  -8, -16,
])
ROOK_MG = _flip([
      6,   8,  10,  12,  12,  10,   8,   6,
     14,  20,  22,  24,  24,  22,  20,  14,
     -2,   4,   6,   8,   8,   6,   4,  -2,
     -6,   0,   2,   4,   4,   2,   0,  -6,
     -8,  -2,   0,   2,   2,   0,  -2,  -8,
    -10,  -2,   0,   2,   2,   0,  -2, -10,
    -12,  -2,   2,   4,   4,   2,  -2, -12,
     -8,  -6,   4,  10,  10,   6,  -6,  -8,
])
ROOK_EG = _flip([
     14,  12,  12,  10,  10,  12,  12,  14,
     14,  14,  14,  12,  12,  12,  14,  14,
      8,   8,   8,   6,   6,   6,   8,   8,
      4,   4,   6,   4,   4,   4,   4,   4,
      0,   2,   2,   2,   2,   0,   0,   0,
     -4,  -2,  -2,  -2,  -2,  -4,  -4,  -6,
     -8,  -6,  -4,  -4,  -4,  -6,  -8, -10,
     -6,  -4,  -2,  -4,  -4,  -4,  -6, -12,
])
QUEEN_MG = _flip([
    -18, -10,  -6,  -4,  -4,  -6, -10, -18,
    -10,  -4,   2,   2,   2,   2,  -4, -10,
     -6,   2,   6,   8,   8,   6,   2,  -6,
     -4,   2,   8,  10,  10,   8,   2,  -4,
     -4,   4,   8,  10,  10,   8,   2,  -4,
     -6,   6,   6,   6,   6,   6,   4,  -6,
    -10,  -2,   4,   2,   2,   2,  -2, -10,
    -18, -12,  -8,   0,  -6,  -8, -12, -18,
])
QUEEN_EG = _flip([
    -20, -10,  -6,   0,   0,  -6, -10, -20,
    -10,   4,  10,  14,  14,  10,   4, -10,
     -6,  10,  18,  22,  22,  18,  10,  -6,
      0,  14,  22,  28,  28,  22,  14,   0,
      0,  14,  22,  28,  28,  22,  14,   0,
     -6,  10,  18,  22,  22,  18,  10,  -6,
    -12,   2,   8,  12,  12,   8,   2, -12,
    -22, -14, -10,  -6,  -6, -10, -14, -22,
])
KING_MG = _flip([
    -60, -70, -70, -80, -80, -70, -70, -60,
    -55, -65, -65, -75, -75, -65, -65, -55,
    -50, -60, -60, -70, -70, -60, -60, -50,
    -45, -55, -60, -70, -70, -60, -55, -45,
    -30, -40, -50, -60, -60, -50, -40, -30,
    -14, -22, -30, -34, -34, -30, -22, -14,
     16,  16,  -6, -14, -14,  -6,  16,  16,
     22,  34,  10, -12,   4, -12,  30,  22,
])
KING_EG = _flip([
    -52, -30, -18, -10, -10, -18, -30, -52,
    -24,  -4,  10,  16,  16,  10,  -4, -24,
    -12,  12,  26,  32,  32,  26,  12, -12,
    -10,  16,  32,  38,  38,  32,  16, -10,
    -12,  14,  30,  36,  36,  30,  14, -12,
    -16,   6,  20,  26,  26,  20,   6, -16,
    -26,  -8,   6,  10,  10,   6,  -8, -26,
    -56, -34, -22, -16, -16, -22, -34, -56,
])
# fmt: on

_pst_mg = np.zeros((2, 7, 64), dtype=np.int32)
_pst_eg = np.zeros((2, 7, 64), dtype=np.int32)
for _piece, (_mg, _eg) in enumerate(
    [
        (PAWN_MG, PAWN_EG),
        (KNIGHT_MG, KNIGHT_EG),
        (BISHOP_MG, BISHOP_EG),
        (ROOK_MG, ROOK_EG),
        (QUEEN_MG, QUEEN_EG),
        (KING_MG, KING_EG),
    ],
    start=1,
):
    for _square in range(64):
        _pst_mg[0, _piece, _square] = _mg[_square]
        _pst_eg[0, _piece, _square] = _eg[_square]
        _pst_mg[1, _piece, _square] = _mg[_square ^ 56]
        _pst_eg[1, _piece, _square] = _eg[_square ^ 56]
PST_MG = _pst_mg
PST_EG = _pst_eg

# mobility is scored per reachable square, counted from a small base so a boxed-in piece is
# punished rather than a very active one being paid twice
MOBILITY_MG = np.array([0, 0, 6, 6, 4, 2, 0], dtype=np.int32)
MOBILITY_EG = np.array([0, 0, 6, 7, 5, 5, 0], dtype=np.int32)
MOBILITY_BASE = np.array([0, 0, 4, 6, 6, 12, 0], dtype=np.int32)

BISHOP_PAIR_MG = 30
BISHOP_PAIR_EG = 48
DOUBLED_MG, DOUBLED_EG = -10, -22
ISOLATED_MG, ISOLATED_EG = -14, -18
BACKWARD_MG, BACKWARD_EG = -8, -10
CONNECTED_MG, CONNECTED_EG = 8, 6
ROOK_OPEN_MG, ROOK_OPEN_EG = 26, 12
ROOK_SEMI_MG, ROOK_SEMI_EG = 12, 6
ROOK_SEVENTH_MG, ROOK_SEVENTH_EG = 18, 30
KNIGHT_OUTPOST = 18
TEMPO = 12

PASSED_MG = np.array([0, 4, 8, 16, 34, 62, 100, 0], dtype=np.int32)
PASSED_EG = np.array([0, 10, 18, 34, 62, 108, 168, 0], dtype=np.int32)

# king safety: attack units are turned into centipawns through this curve
KING_ATTACK_WEIGHT = np.array([0, 0, 8, 8, 12, 20, 0], dtype=np.int32)
SAFETY_TABLE = np.array(
    [min(500, (units * units) // 5) for units in range(64)], dtype=np.int32
)
SHIELD_MISSING = 14
OPEN_FILE_ON_KING = 20

_file = np.zeros(8, dtype=np.int64)
for _f in range(8):
    _file[_f] = i64(0x0101010101010101 << _f)
FILE_MASK = _file

_adjacent = np.zeros(8, dtype=np.int64)
for _f in range(8):
    _mask = 0
    if _f > 0:
        _mask |= int(_file[_f - 1]) & ((1 << 64) - 1)
    if _f < 7:
        _mask |= int(_file[_f + 1]) & ((1 << 64) - 1)
    _adjacent[_f] = i64(_mask)
ADJACENT_FILES = _adjacent

_front = np.zeros((2, 64), dtype=np.int64)
_passed = np.zeros((2, 64), dtype=np.int64)
_shield = np.zeros((2, 64), dtype=np.int64)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    _ahead_white = 0
    _ahead_black = 0
    for _rr in range(_r + 1, 8):
        _ahead_white |= 1 << (_rr * 8 + _f)
    for _rr in range(0, _r):
        _ahead_black |= 1 << (_rr * 8 + _f)
    _front[0, _sq] = i64(_ahead_white)
    _front[1, _sq] = i64(_ahead_black)
    _span_white = _ahead_white
    _span_black = _ahead_black
    for _df in (-1, 1):
        _nf = _f + _df
        if 0 <= _nf < 8:
            for _rr in range(_r + 1, 8):
                _span_white |= 1 << (_rr * 8 + _nf)
            for _rr in range(0, _r):
                _span_black |= 1 << (_rr * 8 + _nf)
    _passed[0, _sq] = i64(_span_white)
    _passed[1, _sq] = i64(_span_black)
    _shield_white = 0
    _shield_black = 0
    for _df in (-1, 0, 1):
        _nf = _f + _df
        if 0 <= _nf < 8:
            for _step in (1, 2):
                if _r + _step < 8:
                    _shield_white |= 1 << ((_r + _step) * 8 + _nf)
                if _r - _step >= 0:
                    _shield_black |= 1 << ((_r - _step) * 8 + _nf)
    _shield[0, _sq] = i64(_shield_white)
    _shield[1, _sq] = i64(_shield_black)
FRONT_SPAN = _front
PASSED_SPAN = _passed
SHIELD_ZONE = _shield

_zone = np.zeros(64, dtype=np.int64)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    _r = min(max(_r, 1), 6)
    _f = min(max(_f, 1), 6)
    _centre = _r * 8 + _f
    _zone[_sq] = i64((int(KING_ATTACKS[_centre]) & ((1 << 64) - 1)) | (1 << _centre))
KING_ZONE = _zone

_distance = np.zeros((64, 64), dtype=np.int32)
for _a in range(64):
    for _b in range(64):
        _distance[_a, _b] = max(abs((_a >> 3) - (_b >> 3)), abs((_a & 7) - (_b & 7)))
DISTANCE = _distance

_edge = np.zeros(64, dtype=np.int32)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    _edge[_sq] = (3 - min(_r, 7 - _r)) + (3 - min(_f, 7 - _f))
EDGE_PENALTY = _edge

RANK_7 = i64(0x00FF000000000000)
RANK_2 = i64(0x000000000000FF00)
CENTRE = i64(0x0000001818000000)


@njit(cache=False)
def pawn_attack_span(pawns, colour):  # type: ignore[no-untyped-def]
    if colour == 0:
        return ((pawns & NOT_FILE_A) << 7) | ((pawns & NOT_FILE_H) << 9)
    return srl(pawns & NOT_FILE_A, 9) | srl(pawns & NOT_FILE_H, 7)


@njit
def phase_of(bb):  # type: ignore[no-untyped-def]
    total = 0
    for piece in range(2, 6):
        total += PHASE_WEIGHT[piece] * popcount(bb[piece + 1])
    return total if total < TOTAL_PHASE else TOTAL_PHASE


@njit
def evaluate(bb, st):  # type: ignore[no-untyped-def]
    """Centipawn score for the side to move."""
    occupancy = bb[0] | bb[1]
    white_pawns = bb[2] & bb[0]
    black_pawns = bb[2] & bb[1]
    white_attacks = pawn_attack_span(white_pawns, 0)
    black_attacks = pawn_attack_span(black_pawns, 1)

    mg = 0
    eg = 0
    for colour in range(2):
        sign = 1 if colour == 0 else -1
        own = bb[colour]
        own_pawns = white_pawns if colour == 0 else black_pawns
        enemy_pawns = black_pawns if colour == 0 else white_pawns
        enemy_pawn_attacks = black_attacks if colour == 0 else white_attacks
        enemy_king = king_square(bb, 1 - colour)
        zone = KING_ZONE[enemy_king]
        attack_units = 0
        attackers = 0

        side_mg = 0
        side_eg = 0

        pieces = own_pawns
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - 1
            side_mg += MATERIAL_MG[1] + PST_MG[colour, 1, square]
            side_eg += MATERIAL_EG[1] + PST_EG[colour, 1, square]
            file_index = square & 7
            if FILE_MASK[file_index] & FRONT_SPAN[colour, square] & own_pawns:
                side_mg += DOUBLED_MG
                side_eg += DOUBLED_EG
            if not (ADJACENT_FILES[file_index] & own_pawns):
                side_mg += ISOLATED_MG
                side_eg += ISOLATED_EG
            elif not (PASSED_SPAN[1 - colour, square] & ADJACENT_FILES[file_index] & own_pawns):
                # no friendly pawn beside or behind on a neighbouring file
                side_mg += BACKWARD_MG
                side_eg += BACKWARD_EG
            if not (PASSED_SPAN[colour, square] & enemy_pawns):
                rank = square >> 3 if colour == 0 else 7 - (square >> 3)
                side_mg += PASSED_MG[rank]
                side_eg += PASSED_EG[rank]
                promotion = (7 * 8 + file_index) if colour == 0 else file_index
                side_eg += 8 * DISTANCE[enemy_king, promotion]

        connected = own_pawns & pawn_attack_span(own_pawns, colour)
        connected_count = popcount(connected)
        side_mg += CONNECTED_MG * connected_count
        side_eg += CONNECTED_EG * connected_count

        pieces = bb[3] & own
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - 1
            side_mg += MATERIAL_MG[2] + PST_MG[colour, 2, square]
            side_eg += MATERIAL_EG[2] + PST_EG[colour, 2, square]
            moves = KNIGHT_ATTACKS[square] & ~own & ~enemy_pawn_attacks
            count = popcount(moves) - MOBILITY_BASE[2]
            side_mg += MOBILITY_MG[2] * count
            side_eg += MOBILITY_EG[2] * count
            if KNIGHT_ATTACKS[square] & zone:
                attack_units += KING_ATTACK_WEIGHT[2]
                attackers += 1
            span = PASSED_SPAN[colour, square] & ADJACENT_FILES[square & 7]
            own_attack = white_attacks if colour == 0 else black_attacks
            if (own_attack & (np.int64(1) << square)) and not (span & enemy_pawns):
                side_mg += KNIGHT_OUTPOST

        bishops = bb[4] & own
        if popcount(bishops) >= 2:
            side_mg += BISHOP_PAIR_MG
            side_eg += BISHOP_PAIR_EG
        pieces = bishops
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - 1
            side_mg += MATERIAL_MG[3] + PST_MG[colour, 3, square]
            side_eg += MATERIAL_EG[3] + PST_EG[colour, 3, square]
            attacks = bishop_attacks(square, occupancy ^ (bb[6] & own))
            moves = attacks & ~own & ~enemy_pawn_attacks
            count = popcount(moves) - MOBILITY_BASE[3]
            side_mg += MOBILITY_MG[3] * count
            side_eg += MOBILITY_EG[3] * count
            if attacks & zone:
                attack_units += KING_ATTACK_WEIGHT[3]
                attackers += 1

        pieces = bb[5] & own
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - 1
            side_mg += MATERIAL_MG[4] + PST_MG[colour, 4, square]
            side_eg += MATERIAL_EG[4] + PST_EG[colour, 4, square]
            attacks = rook_attacks(square, occupancy ^ ((bb[5] | bb[6]) & own))
            moves = attacks & ~own & ~enemy_pawn_attacks
            count = popcount(moves) - MOBILITY_BASE[4]
            side_mg += MOBILITY_MG[4] * count
            side_eg += MOBILITY_EG[4] * count
            file_index = square & 7
            if not (FILE_MASK[file_index] & own_pawns):
                if not (FILE_MASK[file_index] & enemy_pawns):
                    side_mg += ROOK_OPEN_MG
                    side_eg += ROOK_OPEN_EG
                else:
                    side_mg += ROOK_SEMI_MG
                    side_eg += ROOK_SEMI_EG
            seventh = RANK_7 if colour == 0 else RANK_2
            if (np.int64(1) << square) & seventh:
                side_mg += ROOK_SEVENTH_MG
                side_eg += ROOK_SEVENTH_EG
            if attacks & zone:
                attack_units += KING_ATTACK_WEIGHT[4]
                attackers += 1

        pieces = bb[6] & own
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - 1
            side_mg += MATERIAL_MG[5] + PST_MG[colour, 5, square]
            side_eg += MATERIAL_EG[5] + PST_EG[colour, 5, square]
            attacks = queen_attacks(square, occupancy)
            moves = attacks & ~own & ~enemy_pawn_attacks
            count = popcount(moves) - MOBILITY_BASE[5]
            side_mg += MOBILITY_MG[5] * count
            side_eg += MOBILITY_EG[5] * count
            if attacks & zone:
                attack_units += KING_ATTACK_WEIGHT[5]
                attackers += 1

        own_king = king_square(bb, colour)
        side_mg += PST_MG[colour, 6, own_king]
        side_eg += PST_EG[colour, 6, own_king]

        if attackers >= 2:
            index = attack_units if attack_units < 63 else 63
            side_mg += SAFETY_TABLE[index]

        mg += sign * side_mg
        eg += sign * side_eg

    # pawn shelter in front of each king, middlegame only
    for colour in range(2):
        sign = 1 if colour == 0 else -1
        king = king_square(bb, colour)
        own_pawns = white_pawns if colour == 0 else black_pawns
        enemy_pawns = black_pawns if colour == 0 else white_pawns
        shelter = popcount(SHIELD_ZONE[colour, king] & own_pawns)
        missing = 3 - shelter if shelter < 3 else 0
        mg -= sign * SHIELD_MISSING * missing
        king_file = king & 7
        for offset in range(-1, 2):
            neighbour = king_file + offset
            if 0 <= neighbour < 8 and not (FILE_MASK[neighbour] & (own_pawns | enemy_pawns)):
                mg -= sign * OPEN_FILE_ON_KING

    phase = phase_of(bb)
    blended = mg * phase + eg * (TOTAL_PHASE - phase)
    # floor division rounds towards minus infinity, which would make a position score one
    # centipawn differently from its mirror image; truncating towards zero is symmetric
    score = -((-blended) // TOTAL_PHASE) if blended < 0 else blended // TOTAL_PHASE
    score += scale_endgame(bb, mg, eg, white_pawns, black_pawns)
    if st[0] == 1:
        score = -score
    return score + TEMPO


@njit
def scale_endgame(bb, mg, eg, white_pawns, black_pawns):  # type: ignore[no-untyped-def]
    """Push the winning king towards the loser in pawnless endings, and flatten dead draws."""
    white_material = (
        MATERIAL_EG[1] * popcount(white_pawns)
        + MATERIAL_EG[2] * popcount(bb[3] & bb[0])
        + MATERIAL_EG[3] * popcount(bb[4] & bb[0])
        + MATERIAL_EG[4] * popcount(bb[5] & bb[0])
        + MATERIAL_EG[5] * popcount(bb[6] & bb[0])
    )
    black_material = (
        MATERIAL_EG[1] * popcount(black_pawns)
        + MATERIAL_EG[2] * popcount(bb[3] & bb[1])
        + MATERIAL_EG[3] * popcount(bb[4] & bb[1])
        + MATERIAL_EG[4] * popcount(bb[5] & bb[1])
        + MATERIAL_EG[5] * popcount(bb[6] & bb[1])
    )
    advantage = white_material - black_material
    if advantage > 350 and black_pawns == 0:
        strong = 0
    elif advantage < -350 and white_pawns == 0:
        strong = 1
    else:
        return 0
    weak_king = king_square(bb, 1 - strong)
    strong_king = king_square(bb, strong)
    drive = 10 * EDGE_PENALTY[weak_king] + 4 * (7 - DISTANCE[strong_king, weak_king])
    return drive if strong == 0 else -drive


@njit
def insufficient_material(bb):  # type: ignore[no-untyped-def]
    """True for the material draws the referee would claim anyway."""
    if bb[2] | bb[5] | bb[6]:
        return False
    # king against king, or king and a single minor: no mate exists from here
    return popcount(bb[3] | bb[4]) <= 1


# --- NNUE integration point (not yet implemented) ------------------------------------------
#
# `evaluate()` above is what search.py calls at every leaf; swapping in a trained network means
# adding a second evaluator behind the same (bb, st) -> int signature and switching the call
# sites in search.py over once it's validated.
#
# Important constraint specific to this environment: numba's nopython mode cannot call into
# onnxruntime or torch, so a live per-node inference call is not just slow here, it's not
# possible from inside an @njit function at all. The correct shape is the same one a hand-
# written Rust/C++ NNUE uses: train with torch (nnue-pytorch or similar) exactly as already set
# up, then export the trained weights as plain numpy arrays (.npy/.npz) — not as a live
# onnxruntime session — and hand-write the incremental accumulator and forward pass as ordinary
# @njit functions reading those arrays as frozen globals, the same pattern every table in this
# file already uses. onnxruntime is still useful offline, as a correctness check: run the same
# position through the real ONNX model and confirm your numba forward pass agrees, before
# trusting it in search.
