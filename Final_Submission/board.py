"""Bitboard position representation, magic-bitboard attacks, zobrist hashing, move generation.

Everything hot is `@njit`-compiled and operates on plain numpy arrays rather than objects, so
it can be called from other jitted code (search.py, evaluate.py) without breaking numba's
nopython mode. Tables that only need building once (magic attack tables, zobrist random
numbers, knight/king/pawn attack patterns) are built with plain Python loops at import time —
search.py's own docstring calls this out: numba freezes any array it sees as a global into
read-only memory, so these tables become fast, shared, read-only data for every jitted
function below without needing to be threaded through every call.

Board layout, shared with every other module in this engine:
    bb: int64[8]   bb[0]/bb[1] = white/black occupancy, bb[2..8] = pawn..king occupancy
                    (piece type p has occupancy bb[p + 1], matching python-chess's
                    PAWN=1..KING=6 numbering)
    st: int64[6]   st[0] side (0=white, 1=black), st[1] castling rights (1=WK,2=WQ,4=BK,8=BQ),
                    st[2] en-passant target square (-1 if none), st[3] halfmove clock,
                    st[4] zobrist key, st[5] unused
    mb: int8[64]   mailbox: piece type per square (0 = empty), matching bb's numbering

Move encoding, packed into a single int64:
    bits 0-5    from square
    bits 6-11   to square
    bits 12-14  promotion piece type (0 = none, else 2=N 3=B 4=R 5=Q)
    bits 15-16  flag: 0 normal, FLAG_EP en passant capture, FLAG_CASTLE castling
"""

import numpy as np
from numba import njit

from magics import BISHOP_MAGICS, ROOK_MAGICS

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

MOVE_SLOTS = 256
FLAG_EP = 1
FLAG_CASTLE = 2

PROMO_PIECES = (KNIGHT, BISHOP, ROOK, QUEEN)


def i64(x):  # not hot; only used in module-level table construction, plain Python is fine
    """Casts a Python int to numpy int64, correctly reinterpreting any value with bit 63 set as
    the corresponding negative two's-complement value (plain `np.int64(x)` raises OverflowError
    for such values instead of wrapping, which is the wrong behaviour for a bitboard mask)."""
    return np.array([x & ((1 << 64) - 1)], dtype=np.uint64).view(np.int64)[0]


@njit(cache=False)
def bit(square):  # type: ignore[no-untyped-def]
    return np.int64(1) << square


@njit(cache=False)
def srl(x, n):  # type: ignore[no-untyped-def]
    """Logical (unsigned) right shift — plain `>>` on a signed int64 sign-extends, which is
    wrong for a bitboard where the top bit is just square 63, not a sign."""
    return np.int64(np.uint64(x) >> np.uint64(n))


# De Bruijn multiplication bitscan: the classic constant-time "index of the lowest set bit"
# trick. Numba has no portable builtin for this, so it's built once, here, the same way every
# hand-written bitboard engine does it.
_DEBRUIJN64_PY = 0x03F79D71B4CB0A89
_DEBRUIJN64 = np.uint64(_DEBRUIJN64_PY)
_DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _product = ((1 << _i) * _DEBRUIJN64_PY) & ((1 << 64) - 1)
    _DEBRUIJN_INDEX[_product >> 58] = _i
DEBRUIJN_INDEX = _DEBRUIJN_INDEX


@njit(cache=False)
def lsb(bb):  # type: ignore[no-untyped-def]
    u = np.uint64(bb)
    return DEBRUIJN_INDEX[np.int64((u & (-u)) * _DEBRUIJN64) >> np.int64(58) & 63]


@njit(cache=False)
def popcount(x):  # type: ignore[no-untyped-def]
    """SWAR popcount — portable across numpy/numba versions, no dependency on a builtin."""
    u = np.uint64(x)
    u = u - ((u >> np.uint64(1)) & np.uint64(0x5555555555555555))
    u = (u & np.uint64(0x3333333333333333)) + ((u >> np.uint64(2)) & np.uint64(0x3333333333333333))
    u = (u + (u >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((u * np.uint64(0x0101010101010101)) >> np.uint64(56))


def _sq(file, rank):
    return rank * 8 + file


FILE_A = i64(0x0101010101010101)
FILE_H = i64(int(FILE_A) << 7)
NOT_FILE_A = i64(~int(FILE_A) & ((1 << 64) - 1))
NOT_FILE_H = i64(~int(FILE_H) & ((1 << 64) - 1))
RANK_1 = i64(0xFF)
RANK_8 = i64(int(RANK_1) << 56)

# ---------------------------------------------------------------------------------------------
# knight / king / pawn attack tables
# ---------------------------------------------------------------------------------------------

_knight = np.zeros(64, dtype=np.int64)
_king = np.zeros(64, dtype=np.int64)
_pawn = np.zeros((2, 64), dtype=np.int64)
for _sqr in range(64):
    _f0, _r0 = _sqr & 7, _sqr >> 3
    _n = 0
    for _df, _dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        _f, _r = _f0 + _df, _r0 + _dr
        if 0 <= _f < 8 and 0 <= _r < 8:
            _n |= 1 << _sq(_f, _r)
    _knight[_sqr] = i64(_n)
    _k = 0
    for _df in (-1, 0, 1):
        for _dr in (-1, 0, 1):
            if _df == 0 and _dr == 0:
                continue
            _f, _r = _f0 + _df, _r0 + _dr
            if 0 <= _f < 8 and 0 <= _r < 8:
                _k |= 1 << _sq(_f, _r)
    _king[_sqr] = i64(_k)
    _wp, _bp = 0, 0
    for _df in (-1, 1):
        _f = _f0 + _df
        if 0 <= _f < 8:
            if _r0 + 1 < 8:
                _wp |= 1 << _sq(_f, _r0 + 1)
            if _r0 - 1 >= 0:
                _bp |= 1 << _sq(_f, _r0 - 1)
    _pawn[0, _sqr] = i64(_wp)
    _pawn[1, _sqr] = i64(_bp)
KNIGHT_ATTACKS = _knight
KING_ATTACKS = _king
PAWN_ATTACKS = _pawn

# ---------------------------------------------------------------------------------------------
# magic bitboard sliding attacks
# ---------------------------------------------------------------------------------------------

_ROOK_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_DELTAS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _ray_attacks(square, occupied, deltas):
    attacks = 0
    f0, r0 = square & 7, square >> 3
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            s = _sq(f, r)
            attacks |= 1 << s
            if occupied & (1 << s):
                break
            f, r = f + df, r + dr
    return attacks


def _relevant_mask(square, deltas):
    mask = 0
    f0, r0 = square & 7, square >> 3
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            nf, nr = f + df, r + dr
            if not (0 <= nf < 8 and 0 <= nr < 8):
                break
            mask |= 1 << _sq(f, r)
            f, r = f + df, r + dr
    return mask


def _subsets(mask):
    out = []
    subset = 0
    while True:
        out.append(subset)
        subset = (subset - mask) & mask
        if subset == 0:
            break
    return out


def _build_magic_tables(magics, deltas):
    masks = np.zeros(64, dtype=np.int64)
    shifts = np.zeros(64, dtype=np.int64)
    offsets = np.zeros(64, dtype=np.int64)
    table_parts = []
    cursor = 0
    for square in range(64):
        mask = _relevant_mask(square, deltas)
        bits = bin(mask).count("1")
        shift = 64 - bits
        masks[square] = i64(mask)
        shifts[square] = shift
        offsets[square] = cursor
        size = 1 << bits
        entries = [-1] * size  # -1 sentinel: unfilled, used only for the collision check below
        magic = magics[square] & ((1 << 64) - 1)  # magics.py values are already 64-bit patterns
        for occ in _subsets(mask):
            product = (occ * magic) & ((1 << 64) - 1)
            index = product >> shift
            attacks = _ray_attacks(square, occ, deltas)
            if entries[index] != -1 and entries[index] != attacks:
                raise RuntimeError(f"magic collision at square {square}")
            entries[index] = attacks
        table_parts.extend(x if x != -1 else 0 for x in entries)
        cursor += size
    table = np.array([i64(x) for x in table_parts], dtype=np.int64)
    return masks, shifts, offsets, table


_rook_magics_arr = np.array([i64(m) for m in ROOK_MAGICS], dtype=np.int64)
_bishop_magics_arr = np.array([i64(m) for m in BISHOP_MAGICS], dtype=np.int64)

ROOK_MASKS, ROOK_SHIFTS, ROOK_OFFSETS, ROOK_TABLE = _build_magic_tables(ROOK_MAGICS, _ROOK_DELTAS)
BISHOP_MASKS, BISHOP_SHIFTS, BISHOP_OFFSETS, BISHOP_TABLE = _build_magic_tables(BISHOP_MAGICS, _BISHOP_DELTAS)

assert ROOK_TABLE.shape[0] == 102_400, f"rook table size {ROOK_TABLE.shape[0]}, expected 102400"
assert BISHOP_TABLE.shape[0] == 5_248, f"bishop table size {BISHOP_TABLE.shape[0]}, expected 5248"


@njit(cache=False)
def rook_attacks(square, occupancy):  # type: ignore[no-untyped-def]
    occ = occupancy & ROOK_MASKS[square]
    index = srl(occ * _rook_magics_arr[square], ROOK_SHIFTS[square])
    return ROOK_TABLE[ROOK_OFFSETS[square] + index]


@njit(cache=False)
def bishop_attacks(square, occupancy):  # type: ignore[no-untyped-def]
    occ = occupancy & BISHOP_MASKS[square]
    index = srl(occ * _bishop_magics_arr[square], BISHOP_SHIFTS[square])
    return BISHOP_TABLE[BISHOP_OFFSETS[square] + index]


@njit(cache=False)
def queen_attacks(square, occupancy):  # type: ignore[no-untyped-def]
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)


# ---------------------------------------------------------------------------------------------
# zobrist hashing
# ---------------------------------------------------------------------------------------------

_state = np.uint64(0x9E3779B97F4A7C15)


def _next_rand():
    global _state
    _state ^= (_state << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    _state ^= _state >> np.uint64(7)
    _state ^= (_state << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return np.int64(_state)


_zpiece = np.zeros((2, 7, 64), dtype=np.int64)
for _c in range(2):
    for _p in range(1, 7):
        for _s in range(64):
            _zpiece[_c, _p, _s] = _next_rand()
ZOBRIST_PIECE = _zpiece
ZOBRIST_SIDE = _next_rand()
ZOBRIST_CASTLING = np.array([_next_rand() for _ in range(16)], dtype=np.int64)
ZOBRIST_EP_FILE = np.array([_next_rand() for _ in range(8)], dtype=np.int64)


@njit(cache=False)
def zobrist(bb, st):  # type: ignore[no-untyped-def]
    """Full zobrist hash computed from scratch. Used whenever there's no prior position to
    update incrementally from — a fresh FEN handed in by the platform. `make_move`/`make_null`
    maintain st[4] incrementally instead, which must always agree with what this function would
    compute for the same position; that agreement is what lets the transposition table and
    repetition detection work at all."""
    key = np.int64(0)
    for colour in range(2):
        for piece in range(1, 7):
            pieces = bb[piece + 1] & bb[colour]
            while pieces:
                square = lsb(pieces)
                pieces &= pieces - 1
                key ^= ZOBRIST_PIECE[colour, piece, square]
    if st[0] == 1:
        key ^= ZOBRIST_SIDE
    key ^= ZOBRIST_CASTLING[st[1] & 15]
    if st[2] >= 0:
        key ^= ZOBRIST_EP_FILE[st[2] & 7]
    return key


@njit(cache=False)
def pawn_hash(bb):  # type: ignore[no-untyped-def]
    """Hash of pawn placement only (both colours). Used to bucket the search's correction
    history: pawn structure changes slowly and is a decent proxy for "is the static eval's bias
    in this kind of position systematically off," which is what correction history learns and
    compensates for during search."""
    key = np.int64(0)
    for colour in range(2):
        pieces = bb[2] & bb[colour]
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - 1
            key ^= ZOBRIST_PIECE[colour, PAWN, square]
    return key


# ---------------------------------------------------------------------------------------------
# check detection
# ---------------------------------------------------------------------------------------------

@njit(cache=False)
def king_square(bb, colour):  # type: ignore[no-untyped-def]
    return lsb(bb[7] & bb[colour])


@njit(cache=False)
def is_attacked(bb, square, by_colour, occupancy):  # type: ignore[no-untyped-def]
    if PAWN_ATTACKS[1 - by_colour, square] & bb[2] & bb[by_colour]:
        return True
    if KNIGHT_ATTACKS[square] & bb[3] & bb[by_colour]:
        return True
    if KING_ATTACKS[square] & bb[7] & bb[by_colour]:
        return True
    if bishop_attacks(square, occupancy) & (bb[4] | bb[6]) & bb[by_colour]:
        return True
    if rook_attacks(square, occupancy) & (bb[5] | bb[6]) & bb[by_colour]:
        return True
    return False


@njit(cache=False)
def in_check(bb, side):  # type: ignore[no-untyped-def]
    king = king_square(bb, side)
    return is_attacked(bb, king, 1 - side, bb[0] | bb[1])


# ---------------------------------------------------------------------------------------------
# move generation
# ---------------------------------------------------------------------------------------------

@njit(cache=False)
def _encode(frm, to, promo, flag):  # type: ignore[no-untyped-def]
    return frm | (to << 6) | (promo << 12) | (flag << 15)


@njit(cache=False)
def gen_moves(bb, st, mb, moves, start):  # type: ignore[no-untyped-def]
    """Every pseudo-legal move (own-king-safety not checked — the caller applies the move and
    calls in_check to filter, which is cheaper overall than checking legality per candidate)."""
    us = st[0]
    them = 1 - us
    occupied = bb[0] | bb[1]
    own = bb[us]
    enemy = bb[them]
    empty = ~occupied
    idx = start

    push_dir = 8 if us == 0 else -8
    start_rank = 1 if us == 0 else 6
    promo_rank = 7 if us == 0 else 0

    pawns = bb[2] & own
    while pawns:
        frm = lsb(pawns)
        pawns &= pawns - 1
        to = frm + push_dir
        if 0 <= to < 64 and (bit(to) & empty):
            if (to >> 3) == promo_rank:
                for p in range(len(PROMO_PIECES_ARR)):
                    moves[idx] = _encode(frm, to, PROMO_PIECES_ARR[p], 0)
                    idx += 1
            else:
                moves[idx] = _encode(frm, to, 0, 0)
                idx += 1
                if (frm >> 3) == start_rank:
                    to2 = frm + push_dir * 2
                    if bit(to2) & empty:
                        moves[idx] = _encode(frm, to2, 0, 0)
                        idx += 1
        targets = PAWN_ATTACKS[us, frm] & enemy
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            if (to >> 3) == promo_rank:
                for p in range(len(PROMO_PIECES_ARR)):
                    moves[idx] = _encode(frm, to, PROMO_PIECES_ARR[p], 0)
                    idx += 1
            else:
                moves[idx] = _encode(frm, to, 0, 0)
                idx += 1
        if st[2] >= 0 and (PAWN_ATTACKS[us, frm] & bit(st[2])):
            moves[idx] = _encode(frm, st[2], 0, FLAG_EP)
            idx += 1

    knights = bb[3] & own
    while knights:
        frm = lsb(knights)
        knights &= knights - 1
        targets = KNIGHT_ATTACKS[frm] & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    bishops = bb[4] & own
    while bishops:
        frm = lsb(bishops)
        bishops &= bishops - 1
        targets = bishop_attacks(frm, occupied) & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    rooks = bb[5] & own
    while rooks:
        frm = lsb(rooks)
        rooks &= rooks - 1
        targets = rook_attacks(frm, occupied) & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    queens = bb[6] & own
    while queens:
        frm = lsb(queens)
        queens &= queens - 1
        targets = queen_attacks(frm, occupied) & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    king_from = king_square(bb, us)
    targets = KING_ATTACKS[king_from] & ~own
    while targets:
        to = lsb(targets)
        targets &= targets - 1
        moves[idx] = _encode(king_from, to, 0, 0)
        idx += 1

    them_colour = them
    if us == 0:
        if (st[1] & 1) and not (occupied & (bit(5) | bit(6))) \
                and not is_attacked(bb, 4, them_colour, occupied) \
                and not is_attacked(bb, 5, them_colour, occupied) \
                and not is_attacked(bb, 6, them_colour, occupied):
            moves[idx] = _encode(4, 6, 0, FLAG_CASTLE)
            idx += 1
        if (st[1] & 2) and not (occupied & (bit(1) | bit(2) | bit(3))) \
                and not is_attacked(bb, 4, them_colour, occupied) \
                and not is_attacked(bb, 3, them_colour, occupied) \
                and not is_attacked(bb, 2, them_colour, occupied):
            moves[idx] = _encode(4, 2, 0, FLAG_CASTLE)
            idx += 1
    else:
        if (st[1] & 4) and not (occupied & (bit(61) | bit(62))) \
                and not is_attacked(bb, 60, them_colour, occupied) \
                and not is_attacked(bb, 61, them_colour, occupied) \
                and not is_attacked(bb, 62, them_colour, occupied):
            moves[idx] = _encode(60, 62, 0, FLAG_CASTLE)
            idx += 1
        if (st[1] & 8) and not (occupied & (bit(57) | bit(58) | bit(59))) \
                and not is_attacked(bb, 60, them_colour, occupied) \
                and not is_attacked(bb, 59, them_colour, occupied) \
                and not is_attacked(bb, 58, them_colour, occupied):
            moves[idx] = _encode(60, 58, 0, FLAG_CASTLE)
            idx += 1

    return idx


PROMO_PIECES_ARR = np.array(PROMO_PIECES, dtype=np.int64)


@njit(cache=False)
def gen_captures(bb, st, mb, moves, start):  # type: ignore[no-untyped-def]
    """Captures, en passant, and every promotion (capturing or not) — the move set quiescence
    searches, since a pushed promotion is exactly the kind of "loud" move a quiet-position
    evaluation would misjudge if left for the horizon."""
    us = st[0]
    them = 1 - us
    occupied = bb[0] | bb[1]
    own = bb[us]
    enemy = bb[them]
    idx = start
    promo_rank = 7 if us == 0 else 0
    push_dir = 8 if us == 0 else -8

    pawns = bb[2] & own
    while pawns:
        frm = lsb(pawns)
        pawns &= pawns - 1
        to = frm + push_dir
        if 0 <= to < 64 and (bit(to) & ~occupied) and (to >> 3) == promo_rank:
            for p in range(len(PROMO_PIECES_ARR)):
                moves[idx] = _encode(frm, to, PROMO_PIECES_ARR[p], 0)
                idx += 1
        targets = PAWN_ATTACKS[us, frm] & enemy
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            if (to >> 3) == promo_rank:
                for p in range(len(PROMO_PIECES_ARR)):
                    moves[idx] = _encode(frm, to, PROMO_PIECES_ARR[p], 0)
                    idx += 1
            else:
                moves[idx] = _encode(frm, to, 0, 0)
                idx += 1
        if st[2] >= 0 and (PAWN_ATTACKS[us, frm] & bit(st[2])):
            moves[idx] = _encode(frm, st[2], 0, FLAG_EP)
            idx += 1

    knights = bb[3] & own
    while knights:
        frm = lsb(knights)
        knights &= knights - 1
        targets = KNIGHT_ATTACKS[frm] & enemy
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    bishops = bb[4] & own
    while bishops:
        frm = lsb(bishops)
        bishops &= bishops - 1
        targets = bishop_attacks(frm, occupied) & enemy
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    rooks = bb[5] & own
    while rooks:
        frm = lsb(rooks)
        rooks &= rooks - 1
        targets = rook_attacks(frm, occupied) & enemy
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    queens = bb[6] & own
    while queens:
        frm = lsb(queens)
        queens &= queens - 1
        targets = queen_attacks(frm, occupied) & enemy
        while targets:
            to = lsb(targets)
            targets &= targets - 1
            moves[idx] = _encode(frm, to, 0, 0)
            idx += 1

    king_from = king_square(bb, us)
    targets = KING_ATTACKS[king_from] & enemy
    while targets:
        to = lsb(targets)
        targets &= targets - 1
        moves[idx] = _encode(king_from, to, 0, 0)
        idx += 1

    return idx


# ---------------------------------------------------------------------------------------------
# make / unmake
# ---------------------------------------------------------------------------------------------

@njit(cache=False)
def _pawn_could_capture_ep(bb, colour, pawn_square):  # type: ignore[no-untyped-def]
    """Whether a `colour` pawn sits beside `pawn_square` (same rank, adjacent file) — i.e.
    whether an en-passant capture is actually available right now. Only set st[2] when this is
    true, matching position.py's convention, so a position hashes identically whether it was
    reached by search or handed in fresh as a FEN."""
    f = pawn_square & 7
    mask = np.int64(0)
    if f > 0:
        mask |= bit(pawn_square - 1)
    if f < 7:
        mask |= bit(pawn_square + 1)
    return (mask & bb[2] & bb[colour]) != 0


@njit(cache=False)
def make_move(bbs, sts, mbs, ply, move):  # type: ignore[no-untyped-def]
    """Applies `move` at `ply`, writing the result into `ply + 1`. Maintains the zobrist key
    (st[4]) incrementally rather than recomputing it, since this runs on every node of the
    search — the one-time full computation lives in `zobrist()` above, for positions with no
    prior state to update from."""
    bb = bbs[ply].copy()
    st = sts[ply].copy()
    mb = mbs[ply].copy()

    us = st[0]
    them = 1 - us
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 3

    piece = mb[frm]
    key = st[4]

    # remove the mover from its origin square
    bb[us] &= ~bit(frm)
    bb[piece + 1] &= ~bit(frm)
    mb[frm] = 0
    key ^= ZOBRIST_PIECE[us, piece, frm]

    captured_sq = to
    if flag == FLAG_EP:
        captured_sq = to - 8 if us == 0 else to + 8
    captured_piece = mb[captured_sq]
    if captured_piece != 0:
        bb[them] &= ~bit(captured_sq)
        bb[captured_piece + 1] &= ~bit(captured_sq)
        mb[captured_sq] = 0
        key ^= ZOBRIST_PIECE[them, captured_piece, captured_sq]

    placed_piece = promo if promo != 0 else piece
    bb[us] |= bit(to)
    bb[placed_piece + 1] |= bit(to)
    mb[to] = placed_piece
    key ^= ZOBRIST_PIECE[us, placed_piece, to]

    if flag == FLAG_CASTLE:
        if to == 6:
            rook_frm, rook_to = 7, 5
        elif to == 2:
            rook_frm, rook_to = 0, 3
        elif to == 62:
            rook_frm, rook_to = 63, 61
        else:
            rook_frm, rook_to = 56, 59
        bb[us] &= ~bit(rook_frm)
        bb[5] &= ~bit(rook_frm)
        mb[rook_frm] = 0
        key ^= ZOBRIST_PIECE[us, ROOK, rook_frm]
        bb[us] |= bit(rook_to)
        bb[5] |= bit(rook_to)
        mb[rook_to] = ROOK
        key ^= ZOBRIST_PIECE[us, ROOK, rook_to]

    rights = st[1]
    if piece == KING:
        rights &= ~(CASTLE_WK | CASTLE_WQ) if us == 0 else ~(CASTLE_BK | CASTLE_BQ)
    if frm == 0 or captured_sq == 0:
        rights &= ~CASTLE_WQ
    if frm == 7 or captured_sq == 7:
        rights &= ~CASTLE_WK
    if frm == 56 or captured_sq == 56:
        rights &= ~CASTLE_BQ
    if frm == 63 or captured_sq == 63:
        rights &= ~CASTLE_BK
    key ^= ZOBRIST_CASTLING[st[1] & 15]
    key ^= ZOBRIST_CASTLING[rights & 15]

    if st[2] >= 0:
        key ^= ZOBRIST_EP_FILE[st[2] & 7]
    new_ep = np.int64(-1)
    if piece == PAWN and (to - frm == 16 or frm - to == 16):
        candidate = to - 8 if us == 0 else to + 8
        if _pawn_could_capture_ep(bb, them, to):
            new_ep = candidate
            key ^= ZOBRIST_EP_FILE[new_ep & 7]

    key ^= ZOBRIST_SIDE

    sts[ply + 1, 0] = them
    sts[ply + 1, 1] = rights
    sts[ply + 1, 2] = new_ep
    sts[ply + 1, 3] = 0 if (piece == PAWN or captured_piece != 0) else st[3] + 1
    sts[ply + 1, 4] = key
    sts[ply + 1, 5] = 0
    bbs[ply + 1] = bb
    mbs[ply + 1] = mb


@njit(cache=False)
def make_null(bbs, sts, mbs, ply):  # type: ignore[no-untyped-def]
    """A pass move for null-move pruning: flips the side to move, nothing else changes."""
    st = sts[ply]
    key = st[4]
    if st[2] >= 0:
        key ^= ZOBRIST_EP_FILE[st[2] & 7]
    key ^= ZOBRIST_SIDE

    bbs[ply + 1] = bbs[ply]
    mbs[ply + 1] = mbs[ply]
    sts[ply + 1, 0] = 1 - st[0]
    sts[ply + 1, 1] = st[1]
    sts[ply + 1, 2] = -1
    sts[ply + 1, 3] = st[3]
    sts[ply + 1, 4] = key
    sts[ply + 1, 5] = 0
