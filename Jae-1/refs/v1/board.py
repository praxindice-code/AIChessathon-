"""Bitboard position, magic move generation and Zobrist hashing, jitted with numba.

The engine keeps its own position representation rather than searching on python-chess objects,
because python-chess move generation costs orders of magnitude more per node than the jitted
generator here. python-chess is still used at the edges: it parses the FEN we are handed and it
validates the move we hand back.

Layout of a position, all plain numpy arrays so numba can see through them:

    bb   int64[8]   0 white occupancy, 1 black occupancy, then pawn..king at piece type + 1
    st   int64[6]   side to move, castling rights, ep square, halfmove clock, zobrist, spare
    mb   int8[64]   piece type standing on each square, 0 for empty

Squares are in python-chess order: a1 = 0, h8 = 63. Bitboards live in int64, so every logical
right shift goes through srl(); a bare >> would sign-extend bit 63 and corrupt the board.
"""

import numpy as np
from numba import njit

from magics import BISHOP_MAGICS, ROOK_MAGICS

MASK64 = (1 << 64) - 1


def i64(value: int) -> int:
    """Reinterpret a 64 bit pattern as the int64 numba and numpy actually store."""
    value &= MASK64
    return value - (1 << 64) if value >> 63 else value


# --------------------------------------------------------------------------------------------
# constants

WHITE = 0
BLACK = 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

FLAG_NORMAL, FLAG_EP, FLAG_CASTLE, FLAG_DOUBLE = 0, 1, 2, 3

FULL = i64(MASK64)
FILE_A = i64(0x0101010101010101)
FILE_H = i64(0x8080808080808080)
NOT_FILE_A = i64(~0x0101010101010101)
NOT_FILE_H = i64(~0x8080808080808080)
RANK_1 = i64(0x00000000000000FF)
RANK_3 = i64(0x0000000000FF0000)
RANK_6 = i64(0x0000FF0000000000)
RANK_8 = i64(0xFF00000000000000)
PROMO_RANKS = i64(0xFF000000000000FF)

SHIFT_MASK = np.array([i64(MASK64 >> n) for n in range(64)], dtype=np.int64)

DEBRUIJN = i64(0x03F79D71B4CB0A89)
_index = np.zeros(64, dtype=np.int64)
for _bit in range(64):
    _index[((1 << _bit) * 0x03F79D71B4CB0A89 & MASK64) >> 58] = _bit
DEBRUIJN_INDEX = _index

K1 = i64(0x5555555555555555)
K2 = i64(0x3333333333333333)
K4 = i64(0x0F0F0F0F0F0F0F0F)
KF = i64(0x0101010101010101)


@njit(cache=False)
def srl(x, n):  # type: ignore[no-untyped-def]
    """Logical right shift on int64."""
    return (x >> n) & SHIFT_MASK[n]


@njit(cache=False)
def lsb(x):  # type: ignore[no-untyped-def]
    """Index of the least significant set bit; x must be non-zero."""
    return DEBRUIJN_INDEX[srl((x & -x) * DEBRUIJN, 58)]


@njit(cache=False)
def popcount(x):  # type: ignore[no-untyped-def]
    x = x - (srl(x, 1) & K1)
    x = (x & K2) + (srl(x, 2) & K2)
    x = (x + srl(x, 4)) & K4
    return srl(x * KF, 56) & 127


@njit(cache=False)
def bit(square):  # type: ignore[no-untyped-def]
    return np.int64(1) << square


# --------------------------------------------------------------------------------------------
# attack tables

_KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING_STEPS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _steps(square: int, offsets: tuple[tuple[int, int], ...]) -> int:
    rank, file = divmod(square, 8)
    board = 0
    for dr, df in offsets:
        r, f = rank + dr, file + df
        if 0 <= r < 8 and 0 <= f < 8:
            board |= 1 << (r * 8 + f)
    return board


def _ray(square: int, dr: int, df: int, occupancy: int, stop_at_edge: bool) -> int:
    rank, file = divmod(square, 8)
    board = 0
    while True:
        rank += dr
        file += df
        if not (0 <= rank < 8 and 0 <= file < 8):
            break
        if stop_at_edge and (not (0 < rank < 7 or dr == 0) or not (0 < file < 7 or df == 0)):
            break
        target = rank * 8 + file
        board |= 1 << target
        if occupancy >> target & 1:
            break
    return board


def _slider(square: int, occupancy: int, dirs: tuple[tuple[int, int], ...], edge: bool) -> int:
    board = 0
    for dr, df in dirs:
        board |= _ray(square, dr, df, occupancy, edge)
    return board


KNIGHT_ATTACKS = np.array([i64(_steps(s, _KNIGHT_STEPS)) for s in range(64)], dtype=np.int64)
KING_ATTACKS = np.array([i64(_steps(s, _KING_STEPS)) for s in range(64)], dtype=np.int64)
PAWN_ATTACKS = np.array(
    [
        [i64(_steps(s, ((1, -1), (1, 1)))) for s in range(64)],
        [i64(_steps(s, ((-1, -1), (-1, 1)))) for s in range(64)],
    ],
    dtype=np.int64,
)

ROOK_MASK = np.array([i64(_slider(s, 0, _ROOK_DIRS, True)) for s in range(64)], dtype=np.int64)
BISHOP_MASK = np.array([i64(_slider(s, 0, _BISHOP_DIRS, True)) for s in range(64)], dtype=np.int64)
ROOK_MAGIC = np.array(ROOK_MAGICS, dtype=np.int64)
BISHOP_MAGIC = np.array(BISHOP_MAGICS, dtype=np.int64)
ROOK_SHIFT = np.array(
    [64 - bin(int(ROOK_MASK[s]) & MASK64).count("1") for s in range(64)], dtype=np.int64
)
BISHOP_SHIFT = np.array(
    [64 - bin(int(BISHOP_MASK[s]) & MASK64).count("1") for s in range(64)], dtype=np.int64
)


def _offsets(shifts: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros(64, dtype=np.int64)
    total = 0
    for square in range(64):
        out[square] = total
        total += 1 << (64 - int(shifts[square]))
    return out, total


ROOK_OFFSET, _ROOK_TOTAL = _offsets(ROOK_SHIFT)
BISHOP_OFFSET, _BISHOP_TOTAL = _offsets(BISHOP_SHIFT)
ROOK_TABLE = np.zeros(_ROOK_TOTAL, dtype=np.int64)
BISHOP_TABLE = np.zeros(_BISHOP_TOTAL, dtype=np.int64)

_ROOK_DR = np.array([d[0] for d in _ROOK_DIRS], dtype=np.int64)
_ROOK_DF = np.array([d[1] for d in _ROOK_DIRS], dtype=np.int64)
_BISHOP_DR = np.array([d[0] for d in _BISHOP_DIRS], dtype=np.int64)
_BISHOP_DF = np.array([d[1] for d in _BISHOP_DIRS], dtype=np.int64)


@njit(cache=False)
def _fill(masks, magics, shifts, offsets, table, drs, dfs):  # type: ignore[no-untyped-def]
    """Walk every occupancy subset of every square and index it by that square's magic."""
    for square in range(64):
        mask = masks[square]
        occupancy = np.int64(0)
        while True:
            attacks = np.int64(0)
            for direction in range(4):
                rank = square >> 3
                file = square & 7
                while True:
                    rank += drs[direction]
                    file += dfs[direction]
                    if rank < 0 or rank > 7 or file < 0 or file > 7:
                        break
                    target = rank * 8 + file
                    attacks |= np.int64(1) << target
                    if occupancy & (np.int64(1) << target):
                        break
            table[offsets[square] + srl(occupancy * magics[square], shifts[square])] = attacks
            occupancy = (occupancy - mask) & mask
            if occupancy == 0:
                break


_fill(ROOK_MASK, ROOK_MAGIC, ROOK_SHIFT, ROOK_OFFSET, ROOK_TABLE, _ROOK_DR, _ROOK_DF)
_fill(BISHOP_MASK, BISHOP_MAGIC, BISHOP_SHIFT, BISHOP_OFFSET, BISHOP_TABLE, _BISHOP_DR, _BISHOP_DF)


@njit(cache=False)
def rook_attacks(square, occupancy):  # type: ignore[no-untyped-def]
    index = srl((occupancy & ROOK_MASK[square]) * ROOK_MAGIC[square], ROOK_SHIFT[square])
    return ROOK_TABLE[ROOK_OFFSET[square] + index]


@njit(cache=False)
def bishop_attacks(square, occupancy):  # type: ignore[no-untyped-def]
    index = srl((occupancy & BISHOP_MASK[square]) * BISHOP_MAGIC[square], BISHOP_SHIFT[square])
    return BISHOP_TABLE[BISHOP_OFFSET[square] + index]


@njit(cache=False)
def queen_attacks(square, occupancy):  # type: ignore[no-untyped-def]
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)


# squares strictly between two aligned squares, used by the static exchange evaluator
_between = np.zeros((64, 64), dtype=np.int64)
for _a in range(64):
    for _dirs in (_ROOK_DIRS, _BISHOP_DIRS):
        for _dr, _df in _dirs:
            _path = 0
            _r, _f = divmod(_a, 8)
            while True:
                _r += _dr
                _f += _df
                if not (0 <= _r < 8 and 0 <= _f < 8):
                    break
                _between[_a][_r * 8 + _f] = i64(_path)
                _path |= 1 << (_r * 8 + _f)
BETWEEN = _between

CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 13
CASTLE_MASK[4] = 12
CASTLE_MASK[7] = 14
CASTLE_MASK[56] = 7
CASTLE_MASK[60] = 3
CASTLE_MASK[63] = 11

_rng = np.random.default_rng(0x5A9E1)
ZOB_PIECE = _rng.integers(-(2**63), 2**63 - 1, size=(2, 7, 64), dtype=np.int64)
ZOB_CASTLE = _rng.integers(-(2**63), 2**63 - 1, size=16, dtype=np.int64)
ZOB_EP = _rng.integers(-(2**63), 2**63 - 1, size=8, dtype=np.int64)
ZOB_SIDE = i64(int(_rng.integers(-(2**63), 2**63 - 1, dtype=np.int64)))


# --------------------------------------------------------------------------------------------
# queries


@njit(cache=False)
def attacked(bb, square, by, occupancy):  # type: ignore[no-untyped-def]
    """Is `square` attacked by any piece of colour `by`, given this occupancy?"""
    side = bb[by]
    if PAWN_ATTACKS[1 - by][square] & bb[2] & side:
        return True
    if KNIGHT_ATTACKS[square] & bb[3] & side:
        return True
    if KING_ATTACKS[square] & bb[7] & side:
        return True
    if bishop_attacks(square, occupancy) & (bb[4] | bb[6]) & side:
        return True
    return rook_attacks(square, occupancy) & (bb[5] | bb[6]) & side != 0


@njit(cache=False)
def king_square(bb, colour):  # type: ignore[no-untyped-def]
    return lsb(bb[7] & bb[colour])


@njit
def in_check(bb, colour):  # type: ignore[no-untyped-def]
    return attacked(bb, king_square(bb, colour), 1 - colour, bb[0] | bb[1])


@njit
def zobrist(bb, st):  # type: ignore[no-untyped-def]
    key = np.int64(0)
    for colour in range(2):
        for piece in range(1, 7):
            pieces = bb[piece + 1] & bb[colour]
            while pieces:
                square = lsb(pieces)
                pieces &= pieces - 1
                key ^= ZOB_PIECE[colour, piece, square]
    key ^= ZOB_CASTLE[st[1]]
    if st[2] >= 0:
        key ^= ZOB_EP[st[2] & 7]
    if st[0] == 1:
        key ^= ZOB_SIDE
    return key


# --------------------------------------------------------------------------------------------
# move generation
#
# A move is packed into an int32: from | to << 6 | promotion piece << 12 | flag << 15.


@njit(cache=False)
def encode(frm, to, promo, flag):  # type: ignore[no-untyped-def]
    return frm | (to << 6) | (promo << 12) | (flag << 15)


@njit(cache=False)
def move_from(move):  # type: ignore[no-untyped-def]
    return move & 63


@njit(cache=False)
def move_to(move):  # type: ignore[no-untyped-def]
    return (move >> 6) & 63


@njit(cache=False)
def move_promo(move):  # type: ignore[no-untyped-def]
    return (move >> 12) & 7


@njit(cache=False)
def move_flag(move):  # type: ignore[no-untyped-def]
    return (move >> 15) & 3


@njit
def gen_moves(bb, st, mb, moves, start):  # type: ignore[no-untyped-def]
    """Append every pseudo-legal move to `moves` and return the new end index."""
    n = start
    us = st[0]
    them = 1 - us
    own = bb[us]
    opp = bb[them]
    occupancy = own | opp
    empty = ~occupancy

    pawns = bb[2] & own
    if us == 0:
        push1 = (pawns << 8) & empty
        push2 = ((push1 & RANK_3) << 8) & empty
        left = ((pawns & NOT_FILE_A) << 7) & opp
        right = ((pawns & NOT_FILE_H) << 9) & opp
        forward = 8
        dleft = 7
        dright = 9
    else:
        push1 = srl(pawns, 8) & empty
        push2 = srl(push1 & RANK_6, 8) & empty
        left = srl(pawns & NOT_FILE_H, 7) & opp
        right = srl(pawns & NOT_FILE_A, 9) & opp
        forward = -8
        dleft = -7
        dright = -9

    for group in range(3):
        if group == 0:
            rest = push1
            delta = forward
        elif group == 1:
            rest = left
            delta = dleft
        else:
            rest = right
            delta = dright
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            frm = to - delta
            if bit(to) & PROMO_RANKS:
                moves[n] = encode(frm, to, QUEEN, FLAG_NORMAL)
                moves[n + 1] = encode(frm, to, KNIGHT, FLAG_NORMAL)
                moves[n + 2] = encode(frm, to, ROOK, FLAG_NORMAL)
                moves[n + 3] = encode(frm, to, BISHOP, FLAG_NORMAL)
                n += 4
            else:
                moves[n] = encode(frm, to, 0, FLAG_NORMAL)
                n += 1
    rest = push2
    while rest:
        to = lsb(rest)
        rest &= rest - 1
        moves[n] = encode(to - 2 * forward, to, 0, FLAG_DOUBLE)
        n += 1
    if st[2] >= 0:
        rest = PAWN_ATTACKS[them][st[2]] & pawns
        while rest:
            frm = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, st[2], 0, FLAG_EP)
            n += 1

    pieces = bb[3] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = KNIGHT_ATTACKS[frm] & ~own
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1
    pieces = bb[4] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = bishop_attacks(frm, occupancy) & ~own
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1
    pieces = bb[5] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = rook_attacks(frm, occupancy) & ~own
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1
    pieces = bb[6] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = queen_attacks(frm, occupancy) & ~own
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1

    king = king_square(bb, us)
    rest = KING_ATTACKS[king] & ~own
    while rest:
        to = lsb(rest)
        rest &= rest - 1
        moves[n] = encode(king, to, 0, FLAG_NORMAL)
        n += 1

    rights = st[1]
    if us == 0:
        if (rights & 1) and not (occupancy & np.int64(0x60)) and not attacked(bb, 4, 1, occupancy):
            if not attacked(bb, 5, 1, occupancy) and not attacked(bb, 6, 1, occupancy):
                moves[n] = encode(4, 6, 0, FLAG_CASTLE)
                n += 1
        if (rights & 2) and not (occupancy & np.int64(0xE)) and not attacked(bb, 4, 1, occupancy):
            if not attacked(bb, 3, 1, occupancy) and not attacked(bb, 2, 1, occupancy):
                moves[n] = encode(4, 2, 0, FLAG_CASTLE)
                n += 1
    else:
        empty_ks = occupancy & np.int64(0x6000000000000000)
        if (rights & 4) and not empty_ks and not attacked(bb, 60, 0, occupancy):
            if not attacked(bb, 61, 0, occupancy) and not attacked(bb, 62, 0, occupancy):
                moves[n] = encode(60, 62, 0, FLAG_CASTLE)
                n += 1
        empty_qs = occupancy & np.int64(0x0E00000000000000)
        if (rights & 8) and not empty_qs and not attacked(bb, 60, 0, occupancy):
            if not attacked(bb, 59, 0, occupancy) and not attacked(bb, 58, 0, occupancy):
                moves[n] = encode(60, 58, 0, FLAG_CASTLE)
                n += 1
    return n


@njit
def gen_captures(bb, st, mb, moves, start):  # type: ignore[no-untyped-def]
    """Captures, en passant and promotions only: the quiescence move set."""
    n = start
    us = st[0]
    them = 1 - us
    own = bb[us]
    opp = bb[them]
    occupancy = own | opp
    empty = ~occupancy

    pawns = bb[2] & own
    if us == 0:
        left = ((pawns & NOT_FILE_A) << 7) & opp
        right = ((pawns & NOT_FILE_H) << 9) & opp
        promo_push = (pawns << 8) & empty & RANK_8
        forward = 8
        dleft = 7
        dright = 9
    else:
        left = srl(pawns & NOT_FILE_H, 7) & opp
        right = srl(pawns & NOT_FILE_A, 9) & opp
        promo_push = srl(pawns, 8) & empty & RANK_1
        forward = -8
        dleft = -7
        dright = -9

    for group in range(2):
        if group == 0:
            rest = left
            delta = dleft
        else:
            rest = right
            delta = dright
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            frm = to - delta
            if bit(to) & PROMO_RANKS:
                moves[n] = encode(frm, to, QUEEN, FLAG_NORMAL)
                moves[n + 1] = encode(frm, to, KNIGHT, FLAG_NORMAL)
                n += 2
            else:
                moves[n] = encode(frm, to, 0, FLAG_NORMAL)
                n += 1
    rest = promo_push
    while rest:
        to = lsb(rest)
        rest &= rest - 1
        moves[n] = encode(to - forward, to, QUEEN, FLAG_NORMAL)
        moves[n + 1] = encode(to - forward, to, KNIGHT, FLAG_NORMAL)
        n += 2
    if st[2] >= 0:
        rest = PAWN_ATTACKS[them][st[2]] & pawns
        while rest:
            frm = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, st[2], 0, FLAG_EP)
            n += 1

    pieces = bb[3] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = KNIGHT_ATTACKS[frm] & opp
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1
    pieces = bb[4] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = bishop_attacks(frm, occupancy) & opp
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1
    pieces = bb[5] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = rook_attacks(frm, occupancy) & opp
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1
    pieces = bb[6] & own
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - 1
        rest = queen_attacks(frm, occupancy) & opp
        while rest:
            to = lsb(rest)
            rest &= rest - 1
            moves[n] = encode(frm, to, 0, FLAG_NORMAL)
            n += 1

    king = king_square(bb, us)
    rest = KING_ATTACKS[king] & opp
    while rest:
        to = lsb(rest)
        rest &= rest - 1
        moves[n] = encode(king, to, 0, FLAG_NORMAL)
        n += 1
    return n


@njit
def make_move(bbs, sts, mbs, ply, move):  # type: ignore[no-untyped-def]
    """Copy the position at `ply` into `ply + 1` and apply `move` there."""
    for i in range(8):
        bbs[ply + 1, i] = bbs[ply, i]
    for i in range(6):
        sts[ply + 1, i] = sts[ply, i]
    for i in range(64):
        mbs[ply + 1, i] = mbs[ply, i]

    bb = bbs[ply + 1]
    st = sts[ply + 1]
    mb = mbs[ply + 1]

    us = st[0]
    them = 1 - us
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 3
    piece = mb[frm]
    key = st[4]

    if st[2] >= 0:
        key ^= ZOB_EP[st[2] & 7]
    key ^= ZOB_CASTLE[st[1]]
    st[3] += 1

    if flag == FLAG_EP:
        captured_square = to - 8 if us == 0 else to + 8
        square_bit = bit(captured_square)
        bb[them] ^= square_bit
        bb[2] ^= square_bit
        mb[captured_square] = 0
        key ^= ZOB_PIECE[them, PAWN, captured_square]
        st[3] = 0
    else:
        captured = mb[to]
        if captured != 0:
            square_bit = bit(to)
            bb[them] ^= square_bit
            bb[captured + 1] ^= square_bit
            key ^= ZOB_PIECE[them, captured, to]
            st[3] = 0

    bb[us] ^= bit(frm) | bit(to)
    bb[piece + 1] ^= bit(frm)
    mb[frm] = 0
    key ^= ZOB_PIECE[us, piece, frm]
    if promo != 0:
        bb[promo + 1] |= bit(to)
        mb[to] = promo
        key ^= ZOB_PIECE[us, promo, to]
    else:
        bb[piece + 1] |= bit(to)
        mb[to] = piece
        key ^= ZOB_PIECE[us, piece, to]
    if piece == PAWN:
        st[3] = 0

    if flag == FLAG_CASTLE:
        if to == 6:
            rook_from = 7
            rook_to = 5
        elif to == 2:
            rook_from = 0
            rook_to = 3
        elif to == 62:
            rook_from = 63
            rook_to = 61
        else:
            rook_from = 56
            rook_to = 59
        swap = bit(rook_from) | bit(rook_to)
        bb[us] ^= swap
        bb[ROOK + 1] ^= swap
        mb[rook_from] = 0
        mb[rook_to] = ROOK
        key ^= ZOB_PIECE[us, ROOK, rook_from] ^ ZOB_PIECE[us, ROOK, rook_to]

    st[2] = -1
    if flag == FLAG_DOUBLE:
        target = (frm + to) // 2
        if PAWN_ATTACKS[us][target] & bb[2] & bb[them]:
            st[2] = target
            key ^= ZOB_EP[target & 7]

    st[1] &= CASTLE_MASK[frm] & CASTLE_MASK[to]
    key ^= ZOB_CASTLE[st[1]]
    key ^= ZOB_SIDE
    st[0] = them
    st[4] = key


@njit
def make_null(bbs, sts, mbs, ply):  # type: ignore[no-untyped-def]
    """Pass the move to the opponent, for null move pruning."""
    for i in range(8):
        bbs[ply + 1, i] = bbs[ply, i]
    for i in range(6):
        sts[ply + 1, i] = sts[ply, i]
    for i in range(64):
        mbs[ply + 1, i] = mbs[ply, i]
    st = sts[ply + 1]
    key = st[4]
    if st[2] >= 0:
        key ^= ZOB_EP[st[2] & 7]
        st[2] = -1
    key ^= ZOB_SIDE
    st[0] = 1 - st[0]
    st[3] += 1
    st[4] = key


@njit
def perft(bbs, sts, mbs, moves, ply, depth):  # type: ignore[no-untyped-def]
    """Leaf count. Only the test suite calls it, to prove the generator against python-chess."""
    start = ply * 256
    end = gen_moves(bbs[ply], sts[ply], mbs[ply], moves, start)
    us = sts[ply, 0]
    total = 0
    for index in range(start, end):
        make_move(bbs, sts, mbs, ply, moves[index])
        if not in_check(bbs[ply + 1], us):
            total += 1 if depth == 1 else perft(bbs, sts, mbs, moves, ply + 1, depth - 1)
    return total
