"""Iterative-deepening alpha-beta search, jitted end to end.

numba freezes any array it sees as a global into read-only memory, so every mutable buffer the
search needs is passed in instead. They are packed into two arrays to keep the recursive call
cheap: `mem`, a small arena holding the move stacks, history, killers, path and control block,
and `tt`, the transposition table. agent.py allocates both once at import and hands the same
two arrays to every search in the game, which is what lets the table and the history survive
from one move to the next.

The shape of the search is conventional and deliberately so:

    iterative deepening with aspiration windows
      -> principal variation search with a transposition table
         -> null move pruning, reverse futility, razoring, late move reductions
            -> quiescence on captures and promotions, with SEE and delta pruning

Time is checked against a real clock every 1024 nodes through numba's objmode. A read costs
about a third of a microsecond, so well under a millisecond per second of search, and it means
the abort is driven by wall time rather than by a node estimate a slow position could blow.
"""

import time

import numpy as np
from numba import njit, objmode

import nnue as nnue_mod
from board import (
    FLAG_CASTLE,
    FLAG_EP,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    MOVE_SLOTS,
    PAWN_ATTACKS,
    bishop_attacks,
    bit,
    gen_captures,
    gen_moves,
    in_check,
    make_move,
    make_null,
    pawn_hash,
    popcount,
    rook_attacks,
)
from evaluate import MATE, MATE_IN_MAX, evaluate, insufficient_material

MAXPLY = 96
STACK = MAXPLY + 8
INF = 1 << 20
GAME_KEYS = 2048

TT_BITS = 23
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
TT_WORDS = TT_SIZE * 2
TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3
TT_SCORE_BIAS = 40000

MAX_HISTORY = 1 << 15

# layout of the `mem` arena, in int64 words
OFF_MOVES = 0
OFF_SCORES = OFF_MOVES + STACK * MOVE_SLOTS
OFF_SEE = OFF_SCORES + STACK * MOVE_SLOTS
OFF_KILLERS = OFF_SEE + STACK * 48
OFF_HISTORY = OFF_KILLERS + STACK * 2
OFF_COUNTER = OFF_HISTORY + 2 * 64 * 64
OFF_CAPTURE_HISTORY = OFF_COUNTER + 2 * 64 * 64
CAPTURE_HISTORY_SIZE = 2 * 7 * 7  # side * moved piece type (0-6) * captured piece type (0-6)
OFF_CONT_HISTORY = OFF_CAPTURE_HISTORY + CAPTURE_HISTORY_SIZE
# 2-ply continuation history ("follow-up history"): indexed by side, the destination square of
# the move played 2 plies ago (by the same side), and the destination square of this move.
CONT_HISTORY_SIZE = 2 * 64 * 64
OFF_ROOT_MOVES = OFF_CONT_HISTORY + CONT_HISTORY_SIZE
OFF_ROOT_SCORES = OFF_ROOT_MOVES + 256
OFF_CTL = OFF_ROOT_SCORES + 256
OFF_PATH = OFF_CTL + 32
CORR_BITS = 16
CORR_SIZE = 1 << CORR_BITS
CORR_MASK = CORR_SIZE - 1
CORR_MAX = 512  # centipawns; a correction bigger than this is almost certainly noise
CORR_SCALE = 256  # fixed-point scale so small per-node updates don't round to zero
OFF_CORRECTION = OFF_PATH + GAME_KEYS + STACK
MEM_WORDS = OFF_CORRECTION + 2 * CORR_SIZE
# converts NNUE's raw (normalized) output into the same integer centipawn-ish scale evaluate()
# already uses — must match the nnue-pytorch trainer's own default `nnue2score` (600.0); if a
# checkpoint was ever trained with a different value, this needs to match that instead
NNUE2SCORE = 600.0

# control block, indices relative to OFF_CTL
CTL_STOP = 0
CTL_NODES = 1
CTL_ROOT_COUNT = 2
CTL_ROOT_MOVE = 3
CTL_ROOT_DONE = 4
CTL_BASE = 5
CTL_BEST_MOVE = 6
CTL_BEST_SCORE = 7
CTL_DEPTH = 8
CTL_HARD_NS = 9
CTL_SOFT_NS = 10
CTL_STABLE = 11

SEE_VALUE = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int64)

_lmr = np.zeros((64, 64), dtype=np.int64)
for _depth in range(1, 64):
    for _index in range(1, 64):
        _lmr[_depth, _index] = int(0.80 + np.log(_depth) * np.log(_index) / 2.30)
LMR = _lmr

FUTILITY = np.array([0, 110, 210, 320, 450, 600, 780, 980], dtype=np.int64)
LATE_MOVE_COUNT = np.array([0, 6, 9, 14, 21, 30, 41, 54], dtype=np.int64)


def new_memory() -> np.ndarray:
    return np.zeros(MEM_WORDS, dtype=np.int64)


def new_table() -> np.ndarray:
    return np.zeros(TT_WORDS, dtype=np.int64)


@njit(cache=False)
def now_ns():  # type: ignore[no-untyped-def]
    with objmode(value="f8"):
        value = time.perf_counter()
    return np.int64(value * 1e9)


@njit(cache=False)
def out_of_time(mem):  # type: ignore[no-untyped-def]
    """Count a node and, once every 1024 of them, ask the clock whether we are done."""
    mem[OFF_CTL + CTL_NODES] += 1
    if mem[OFF_CTL + CTL_STOP] != 0:
        return True
    if mem[OFF_CTL + CTL_NODES] & 1023 != 0:
        return False
    if now_ns() >= mem[OFF_CTL + CTL_HARD_NS]:
        mem[OFF_CTL + CTL_STOP] = 1
        return True
    return False


@njit(cache=False)
def attackers_to(bb, square, occupancy):  # type: ignore[no-untyped-def]
    return (
        (PAWN_ATTACKS[1][square] & bb[2] & bb[0])
        | (PAWN_ATTACKS[0][square] & bb[2] & bb[1])
        | (KNIGHT_ATTACKS[square] & bb[3])
        | (KING_ATTACKS[square] & bb[7])
        | (bishop_attacks(square, occupancy) & (bb[4] | bb[6]))
        | (rook_attacks(square, occupancy) & (bb[5] | bb[6]))
    ) & occupancy


@njit(cache=False)
def see(bbs, mbs, mem, ply, move):  # type: ignore[no-untyped-def]
    """Static exchange evaluation: material won by this capture if both sides trade optimally."""
    bb = bbs[ply]
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 3
    if flag == FLAG_CASTLE:
        return 0
    us = 0 if bb[0] & bit(frm) else 1
    gain = OFF_SEE + ply * 48
    if flag == FLAG_EP:
        mem[gain] = SEE_VALUE[1]
    else:
        mem[gain] = SEE_VALUE[mbs[ply, to]]
    piece = mbs[ply, frm]
    if promo != 0:
        mem[gain] += SEE_VALUE[promo] - SEE_VALUE[1]
        piece = promo
    occupancy = (bb[0] | bb[1]) ^ bit(frm)
    if flag == FLAG_EP:
        occupancy ^= bit(to - 8 if us == 0 else to + 8)

    side = 1 - us
    depth = 0
    while depth < 46:
        depth += 1
        mem[gain + depth] = SEE_VALUE[piece] - mem[gain + depth - 1]
        attackers = attackers_to(bb, to, occupancy) & bb[side]
        if attackers == 0:
            break
        found = False
        for candidate in range(1, 7):
            subset = attackers & bb[candidate + 1]
            if subset:
                occupancy ^= subset & -subset
                piece = candidate
                found = True
                break
        if not found:
            break
        side = 1 - side
    while depth > 1:
        depth -= 1
        higher = -mem[gain + depth - 1]
        if mem[gain + depth] > higher:
            higher = mem[gain + depth]
        mem[gain + depth - 1] = -higher
    return mem[gain]


@njit(cache=False)
def score_moves(bbs, mbs, sts, mem, ply, start, end, tt_move, previous, previous2):  # type: ignore[no-untyped-def]
    """Assign an ordering key to every generated move."""
    side = sts[ply, 0]
    killer_a = mem[OFF_KILLERS + ply * 2]
    killer_b = mem[OFF_KILLERS + ply * 2 + 1]
    counter = 0
    if previous != 0:
        counter = mem[OFF_COUNTER + side * 4096 + (previous & 63) * 64 + ((previous >> 6) & 63)]
    prev2_to = (previous2 >> 6) & 63 if previous2 != 0 else -1
    for index in range(start, end):
        move = mem[OFF_MOVES + index]
        if move == tt_move:
            mem[OFF_SCORES + index] = 1 << 30
            continue
        to = (move >> 6) & 63
        promo = (move >> 12) & 7
        flag = (move >> 15) & 3
        victim = SEE_VALUE[1] if flag == FLAG_EP else SEE_VALUE[mbs[ply, to]]
        if victim != 0 or promo != 0:
            attacker = SEE_VALUE[mbs[ply, move & 63]]
            value = victim * 16 - attacker // 16 + SEE_VALUE[promo] * 8
            if victim != 0:
                moved_piece = mbs[ply, move & 63]
                captured_piece = 1 if flag == FLAG_EP else mbs[ply, to]
                cap_hist = mem[OFF_CAPTURE_HISTORY + side * 49 + moved_piece * 7 + captured_piece]
                value += cap_hist
            if victim != 0 and see(bbs, mbs, mem, ply, move) < 0:
                mem[OFF_SCORES + index] = -(1 << 24) + value
            else:
                mem[OFF_SCORES + index] = (1 << 26) + value
        elif move == killer_a:
            mem[OFF_SCORES + index] = (1 << 25) + 200
        elif move == killer_b:
            mem[OFF_SCORES + index] = (1 << 25) + 100
        elif move == counter:
            mem[OFF_SCORES + index] = 1 << 25
        else:
            score = mem[OFF_HISTORY + side * 4096 + (move & 63) * 64 + to]
            if prev2_to >= 0:
                score += mem[OFF_CONT_HISTORY + side * 4096 + prev2_to * 64 + to]
            mem[OFF_SCORES + index] = score


@njit(cache=False)
def pick_move(mem, current, end):  # type: ignore[no-untyped-def]
    """Selection sort one move at a time: a beta cutoff usually makes the rest wasted work."""
    best = current
    for index in range(current + 1, end):
        if mem[OFF_SCORES + index] > mem[OFF_SCORES + best]:
            best = index
    if best != current:
        swap = mem[OFF_MOVES + current]
        mem[OFF_MOVES + current] = mem[OFF_MOVES + best]
        mem[OFF_MOVES + best] = swap
        swap = mem[OFF_SCORES + current]
        mem[OFF_SCORES + current] = mem[OFF_SCORES + best]
        mem[OFF_SCORES + best] = swap
    return mem[OFF_MOVES + current]


@njit(cache=False)
def repeated(sts, mem, ply):  # type: ignore[no-untyped-def]
    """Has this exact position occurred before, in the search path or earlier in the game?"""
    key = sts[ply, 4]
    limit = sts[ply, 3]
    index = OFF_PATH + mem[OFF_CTL + CTL_BASE] + ply
    back = 2
    while back <= limit and index - back >= OFF_PATH:
        if mem[index - back] == key:
            return True
        back += 2
    return False


@njit(cache=False)
def update_history(mem, ply, move, depth, side):  # type: ignore[no-untyped-def]
    frm = move & 63
    to = (move >> 6) & 63
    slot = OFF_HISTORY + side * 4096 + frm * 64 + to
    mem[slot] += depth * depth
    if mem[slot] > MAX_HISTORY:
        for index in range(OFF_HISTORY, OFF_HISTORY + 8192):
            mem[index] //= 2
    killer = OFF_KILLERS + ply * 2
    if mem[killer] != move:
        mem[killer + 1] = mem[killer]
        mem[killer] = move


@njit(cache=False)
def corr_index(bb, side):  # type: ignore[no-untyped-def]
    bucket = pawn_hash(bb) & CORR_MASK
    return OFF_CORRECTION + side * CORR_SIZE + bucket


@njit(cache=False)
def corrected_eval(bb, st, mem, ply, acc, psqt):  # type: ignore[no-untyped-def]
    """Static eval, adjusted by whatever this pawn structure's running correction currently
    is. The base evaluator is whichever is available: NNUE if weights.npz was found at import
    (see nnue.py), else evaluate.py's hand-tuned formula. Either way it's systematically wrong
    in predictable ways; this nudges it toward what search has actually been finding for
    similar pawn skeletons, at near-zero cost relative to the search that produced the
    correction in the first place."""
    if nnue_mod.NNUE_AVAILABLE:
        piece_count = popcount(bb[0] | bb[1])
        raw = np.int64(nnue_mod.nnue_eval(acc, psqt, st[0], piece_count, ply) * NNUE2SCORE)
    else:
        raw = evaluate(bb, st)
    idx = corr_index(bb, st[0])
    return raw + mem[idx] // CORR_SCALE


@njit(cache=False)
def update_correction(mem, bb, st, static_eval, score, depth):  # type: ignore[no-untyped-def]
    """After a node finishes searching, nudge that pawn structure's correction toward whatever
    gap remained between the (corrected) static eval and what search actually found —
    exponential-moving-average update, weighted more heavily by deeper searches since they're
    more trustworthy signal."""
    if score >= MATE_IN_MAX or score <= -MATE_IN_MAX:
        return  # mate scores aren't a "the eval was off" signal, they're a different thing
    diff = score - static_eval
    if diff > CORR_MAX:
        diff = CORR_MAX
    elif diff < -CORR_MAX:
        diff = -CORR_MAX
    weight = depth + 1
    if weight > 16:
        weight = 16
    idx = corr_index(bb, st[0])
    entry = mem[idx] + (diff * CORR_SCALE - mem[idx]) * weight // (weight + 8)
    cap = CORR_MAX * CORR_SCALE
    if entry > cap:
        entry = cap
    elif entry < -cap:
        entry = -cap
    mem[idx] = entry


@njit("i8(i8[:,::1],i8[:,::1],i1[:,::1],f4[:,:,::1],f4[:,:,::1],i8[::1],i8[::1],i8,i8,i8)", cache=False, nogil=True)
def quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply, alpha, beta):
    """Search captures and promotions until the position is quiet enough to evaluate."""
    if out_of_time(mem):
        return 0
    if ply >= MAXPLY:
        return corrected_eval(bbs[ply], sts[ply], mem, ply, acc, psqt)

    checked = in_check(bbs[ply], sts[ply, 0])
    best = -INF
    if not checked:
        best = corrected_eval(bbs[ply], sts[ply], mem, ply, acc, psqt)
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        if best + 950 < alpha:
            return best

    start = ply * MOVE_SLOTS
    if checked:
        end = gen_moves(bbs[ply], sts[ply], mbs[ply], mem[OFF_MOVES:], start)
    else:
        end = gen_captures(bbs[ply], sts[ply], mbs[ply], mem[OFF_MOVES:], start)
    none = np.int64(0)
    score_moves(bbs, mbs, sts, mem, ply, start, end, none, none, none)

    side = sts[ply, 0]
    base = OFF_PATH + mem[OFF_CTL + CTL_BASE]
    legal = 0
    for current in range(start, end):
        move = pick_move(mem, current, end)
        if not checked:
            if mem[OFF_SCORES + current] < 0:
                break
            to = (move >> 6) & 63
            flag = (move >> 15) & 3
            victim = SEE_VALUE[1] if flag == FLAG_EP else SEE_VALUE[mbs[ply, to]]
            if (move >> 12) & 7 == 0 and best + victim + 200 < alpha:
                continue
        make_move(bbs, sts, mbs, ply, move)
        if nnue_mod.NNUE_AVAILABLE:
            nnue_mod.nnue_make_move(bbs, mbs, acc, psqt, ply)
        if in_check(bbs[ply + 1], side):
            continue
        legal += 1
        mem[base + ply + 1] = sts[ply + 1, 4]
        score = -quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply + 1, -beta, -alpha)
        if mem[OFF_CTL + CTL_STOP] != 0:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if score >= beta:
                    break
    if checked and legal == 0:
        return -MATE + ply
    return best


@njit("i8(i8[:,::1],i8[:,::1],i1[:,::1],f4[:,:,::1],f4[:,:,::1],i8[::1],i8[::1],i8,i8,i8,i8,i8,i8,i8)", cache=False,
      nogil=True)
def negamax(bbs, sts, mbs, acc, psqt, mem, tt, ply, depth, alpha, beta, allow_null, previous, previous2):
    if out_of_time(mem):
        return 0

    pv_node = beta - alpha > 1
    side = sts[ply, 0]
    # np.int64 keeps these out of numba's literal typing, which would recompile negamax
    null_ok = np.int64(1)
    no_move = np.int64(0)
    null_off = np.int64(0)
    base = OFF_PATH + mem[OFF_CTL + CTL_BASE]

    if ply > 0:
        if sts[ply, 3] >= 100 or repeated(sts, mem, ply) or insufficient_material(bbs[ply]):
            return 0
        if ply >= MAXPLY:
            return corrected_eval(bbs[ply], sts[ply], mem, ply, acc, psqt)
        # mate distance pruning: a mate found deeper can never beat one already in hand
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    checked = in_check(bbs[ply], side)
    if checked:
        depth += 1
    if depth <= 0:
        return quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply, alpha, beta)

    key = sts[ply, 4]
    slot = (key & TT_MASK) * 2
    tt_move = 0
    if tt[slot] == key:
        data = tt[slot + 1]
        tt_move = data & 0x1FFFF
        if not pv_node and ((data >> 34) & 0xFF) >= depth:
            stored = ((data >> 17) & 0x1FFFF) - TT_SCORE_BIAS
            if stored > MATE_IN_MAX:
                stored -= ply
            elif stored < -MATE_IN_MAX:
                stored += ply
            flag = (data >> 42) & 3
            if flag == TT_EXACT:
                return stored
            if flag == TT_LOWER and stored >= beta:
                return stored
            if flag == TT_UPPER and stored <= alpha:
                return stored

    static = -INF
    if not checked:
        static = corrected_eval(bbs[ply], sts[ply], mem, ply, acc, psqt)

    if not pv_node and not checked and beta < MATE_IN_MAX and beta > -MATE_IN_MAX:
        # reverse futility: so far above beta that handing over a piece would not bring it below
        if depth <= 7 and static - 85 * depth >= beta:
            return static
        # razoring: hopeless enough that only a tactic saves it, so let quiescence decide
        if depth <= 3 and static + 300 * depth < alpha:
            scouted = quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply, alpha - 1, alpha)
            if mem[OFF_CTL + CTL_STOP] != 0:
                return 0
            if scouted < alpha:
                return scouted
        # null move: give the opponent a free move and see whether we are still winning
        heavy = (bbs[ply, 3] | bbs[ply, 4] | bbs[ply, 5] | bbs[ply, 6]) & bbs[ply, side]
        if allow_null != 0 and depth >= 3 and static >= beta and heavy != 0:
            reduction = 3 + depth // 5
            make_null(bbs, sts, mbs, ply)
            if nnue_mod.NNUE_AVAILABLE:
                nnue_mod.nnue_make_null(acc, psqt, ply)
            mem[base + ply + 1] = sts[ply + 1, 4]
            score = -negamax(
                bbs, sts, mbs, acc, psqt, mem, tt, ply + 1, depth - 1 - reduction,
                -beta, -beta + 1, null_off, no_move, no_move,
            )
            if mem[OFF_CTL + CTL_STOP] != 0:
                return 0
            if score >= beta:
                return beta if score > MATE_IN_MAX else score

    # with no hint from the table, a shallower search is cheaper than a badly ordered one
    if tt_move == 0 and depth >= 5:
        depth -= 1

    start = ply * MOVE_SLOTS
    end = gen_moves(bbs[ply], sts[ply], mbs[ply], mem[OFF_MOVES:], start)
    score_moves(bbs, mbs, sts, mem, ply, start, end, tt_move, previous, previous2)

    mem[OFF_KILLERS + (ply + 2) * 2] = 0
    mem[OFF_KILLERS + (ply + 2) * 2 + 1] = 0

    best = -INF
    best_move = 0
    legal = 0
    original_alpha = alpha
    futile = not checked and not pv_node and depth <= 7 and static + FUTILITY[depth] <= alpha

    for current in range(start, end):
        move = pick_move(mem, current, end)
        to = (move >> 6) & 63
        flag = (move >> 15) & 3
        capture = mbs[ply, to] != 0 or flag == FLAG_EP
        quiet = not capture and (move >> 12) & 7 == 0

        if legal > 0 and best > -MATE_IN_MAX:
            if quiet:
                if futile:
                    continue
                # measured: guarding this with `not pv_node` cost 0.76 ply at a fixed
                # three second budget, which is more than the accuracy it bought back
                if depth <= 7 and legal >= LATE_MOVE_COUNT[depth]:
                    continue
            elif capture and depth <= 6 and see(bbs, mbs, mem, ply, move) < -60 * depth:
                continue

        make_move(bbs, sts, mbs, ply, move)
        if nnue_mod.NNUE_AVAILABLE:
            nnue_mod.nnue_make_move(bbs, mbs, acc, psqt, ply)
        if in_check(bbs[ply + 1], side):
            continue
        legal += 1
        mem[base + ply + 1] = sts[ply + 1, 4]

        reduction = 0
        if depth >= 3 and legal > 2 and quiet and not checked:
            capped_depth = depth if depth < 63 else 63
            capped_index = legal if legal < 63 else 63
            reduction = LMR[capped_depth, capped_index]
            if pv_node:
                reduction -= 1
            if mem[OFF_SCORES + current] > 4000:
                reduction -= 1
            if in_check(bbs[ply + 1], 1 - side):
                reduction -= 1
            if reduction < 0:
                reduction = 0
            if reduction > depth - 2:
                reduction = depth - 2

        if legal == 1:
            score = -negamax(
                bbs, sts, mbs, acc, psqt, mem, tt, ply + 1, depth - 1, -beta, -alpha, null_ok, move, previous
            )
        else:
            score = -negamax(
                bbs, sts, mbs, acc, psqt, mem, tt, ply + 1, depth - 1 - reduction,
                -alpha - 1, -alpha, null_ok, move, previous,
            )
            if score > alpha and reduction > 0:
                score = -negamax(
                    bbs, sts, mbs, acc, psqt, mem, tt, ply + 1, depth - 1, -alpha - 1, -alpha, null_ok, move, previous
                )
            if score > alpha and score < beta:
                score = -negamax(
                    bbs, sts, mbs, acc, psqt, mem, tt, ply + 1, depth - 1, -beta, -alpha, null_ok, move, previous
                )
        if mem[OFF_CTL + CTL_STOP] != 0:
            return 0

        if score > best:
            best = score
            best_move = move
            if score > alpha:
                alpha = score
                if score >= beta:
                    if quiet:
                        update_history(mem, ply, move, depth, side)
                        if previous != 0:
                            counter = (
                                OFF_COUNTER
                                + side * 4096
                                + (previous & 63) * 64
                                + ((previous >> 6) & 63)
                            )
                            mem[counter] = move
                        if previous2 != 0:
                            cont_idx = (
                                OFF_CONT_HISTORY + side * 4096 + ((previous2 >> 6) & 63) * 64 + to
                            )
                            bonus = mem[cont_idx] + depth * depth
                            mem[cont_idx] = bonus if bonus <= MAX_HISTORY else MAX_HISTORY
                    elif capture:
                        moved_piece = mbs[ply, move & 63]
                        captured_piece = 1 if flag == FLAG_EP else mbs[ply, to]
                        cap_idx = OFF_CAPTURE_HISTORY + side * 49 + moved_piece * 7 + captured_piece
                        bonus = mem[cap_idx] + depth * depth
                        mem[cap_idx] = bonus if bonus <= MAX_HISTORY else MAX_HISTORY
                    break

    if legal == 0:
        return -MATE + ply if checked else 0

    stored = best
    if stored > MATE_IN_MAX:
        stored += ply
    elif stored < -MATE_IN_MAX:
        stored -= ply
    flag = TT_EXACT
    if best <= original_alpha:
        flag = TT_UPPER
    elif best >= beta:
        flag = TT_LOWER
    if tt[slot] != key or depth >= ((tt[slot + 1] >> 34) & 0xFF) or flag == TT_EXACT:
        tt[slot] = key
        tt[slot + 1] = (
            (best_move & 0x1FFFF)
            | ((stored + TT_SCORE_BIAS) << 17)
            | (depth << 34)
            | (flag << 42)
        )

    if not checked and depth >= 1:
        update_correction(mem, bbs[ply], sts[ply], static, best, depth)

    return best


@njit(cache=False)
def search_root(bbs, sts, mbs, acc, psqt, mem, tt, depth, alpha, beta):  # type: ignore[no-untyped-def]
    """One pass over the root move list. Leaves the best move in the control block."""
    count = mem[OFF_CTL + CTL_ROOT_COUNT]
    base = OFF_PATH + mem[OFF_CTL + CTL_BASE]
    best = -INF
    null_ok = np.int64(1)
    root = np.int64(0)
    no_prev2 = np.int64(0)
    mem[OFF_CTL + CTL_ROOT_DONE] = 0
    for index in range(count):
        move = mem[OFF_ROOT_MOVES + index]
        make_move(bbs, sts, mbs, root, move)
        if nnue_mod.NNUE_AVAILABLE:
            nnue_mod.nnue_make_move(bbs, mbs, acc, psqt, root)
        mem[base + 1] = sts[1, 4]
        if index == 0:
            score = -negamax(
                bbs, sts, mbs, acc, psqt, mem, tt, 1, depth - 1, -beta, -alpha, null_ok, move, no_prev2
            )
        else:
            score = -negamax(
                bbs, sts, mbs, acc, psqt, mem, tt, 1, depth - 1, -alpha - 1, -alpha, null_ok, move, no_prev2
            )
            if score > alpha and score < beta:
                score = -negamax(
                    bbs, sts, mbs, acc, psqt, mem, tt, 1, depth - 1, -beta, -alpha, null_ok, move, no_prev2
                )
        if mem[OFF_CTL + CTL_STOP] != 0:
            break
        mem[OFF_ROOT_SCORES + index] = score
        if score > best:
            best = score
            mem[OFF_CTL + CTL_ROOT_MOVE] = move
            mem[OFF_CTL + CTL_ROOT_DONE] = 1
            if score > alpha:
                alpha = score
    return best


@njit(cache=False)
def order_root(mem):  # type: ignore[no-untyped-def]
    count = mem[OFF_CTL + CTL_ROOT_COUNT]
    for i in range(count):
        best = i
        for j in range(i + 1, count):
            if mem[OFF_ROOT_SCORES + j] > mem[OFF_ROOT_SCORES + best]:
                best = j
        if best != i:
            swap = mem[OFF_ROOT_MOVES + i]
            mem[OFF_ROOT_MOVES + i] = mem[OFF_ROOT_MOVES + best]
            mem[OFF_ROOT_MOVES + best] = swap
            swap = mem[OFF_ROOT_SCORES + i]
            mem[OFF_ROOT_SCORES + i] = mem[OFF_ROOT_SCORES + best]
            mem[OFF_ROOT_SCORES + best] = swap


@njit(cache=False)
def setup_root(bbs, sts, mbs, mem):  # type: ignore[no-untyped-def]
    """Fill the root move list with legal moves only, and report how many there are."""
    root = np.int64(0)
    end = gen_moves(bbs[0], sts[0], mbs[0], mem[OFF_MOVES:], root)
    side = sts[0, 0]
    count = 0
    for index in range(end):
        move = mem[OFF_MOVES + index]
        make_move(bbs, sts, mbs, root, move)
        if not in_check(bbs[1], side):
            mem[OFF_ROOT_MOVES + count] = move
            mem[OFF_ROOT_SCORES + count] = -INF
            count += 1
    mem[OFF_CTL + CTL_ROOT_COUNT] = count
    return count


# nogil so that pondering on the opponent's clock in a background thread cannot starve the
# main thread of the GIL when their move finally arrives
@njit(cache=False, nogil=True)
def think(bbs, sts, mbs, acc, psqt, mem, tt, max_depth):  # type: ignore[no-untyped-def]
    """Iterative deepening. Leaves the chosen move in the control block and always has one."""
    mem[OFF_CTL + CTL_STOP] = 0
    mem[OFF_CTL + CTL_NODES] = 0
    if nnue_mod.NNUE_AVAILABLE:
        nnue_mod.nnue_init_root(bbs[0], acc, psqt)
    for ply in range(STACK * 2):
        mem[OFF_KILLERS + ply] = 0

    count = setup_root(bbs, sts, mbs, mem)
    mem[OFF_CTL + CTL_BEST_MOVE] = 0
    mem[OFF_CTL + CTL_BEST_SCORE] = 0
    mem[OFF_CTL + CTL_DEPTH] = 0
    if count == 0:
        return 0
    mem[OFF_CTL + CTL_BEST_MOVE] = mem[OFF_ROOT_MOVES]
    if count == 1:
        mem[OFF_CTL + CTL_DEPTH] = 1
        return mem[OFF_ROOT_MOVES]

    best_score = 0
    for depth in range(1, max_depth + 1):
        window = 26
        if depth >= 5 and best_score < MATE_IN_MAX and best_score > -MATE_IN_MAX:
            alpha = best_score - window
            beta = best_score + window
        else:
            alpha = -INF
            beta = INF
        score = 0
        while True:
            score = search_root(bbs, sts, mbs, acc, psqt, mem, tt, depth, alpha, beta)
            if mem[OFF_CTL + CTL_STOP] != 0:
                break
            if score <= alpha:
                window *= 3
                alpha = score - window
                if alpha < -INF:
                    alpha = -INF
            elif score >= beta:
                window *= 3
                beta = score + window
                if beta > INF:
                    beta = INF
            else:
                break

        if mem[OFF_CTL + CTL_STOP] != 0:
            # the iteration was cut short; keep its move only if a root move finished and beat
            # everything before it, which is exactly what CTL_ROOT_DONE records
            if mem[OFF_CTL + CTL_ROOT_DONE] != 0:
                mem[OFF_CTL + CTL_BEST_MOVE] = mem[OFF_CTL + CTL_ROOT_MOVE]
            break

        best_score = score
        settled = mem[OFF_CTL + CTL_ROOT_MOVE] == mem[OFF_CTL + CTL_BEST_MOVE]
        mem[OFF_CTL + CTL_BEST_MOVE] = mem[OFF_CTL + CTL_ROOT_MOVE]
        mem[OFF_CTL + CTL_BEST_SCORE] = score
        mem[OFF_CTL + CTL_DEPTH] = depth
        mem[OFF_CTL + CTL_STABLE] = 1 if settled else 0
        order_root(mem)

        # a mate we are delivering is settled; a mate against us still deserves the search
        # time, because a deeper look often finds the longer defence
        if score > MATE_IN_MAX:
            break
        # when the last iteration changed its mind, the position is not understood yet, so
        # spend part of the gap up to the hard limit rather than moving on the newer guess
        limit = mem[OFF_CTL + CTL_SOFT_NS]
        if not settled and depth >= 5:
            limit += (mem[OFF_CTL + CTL_HARD_NS] - limit) // 3
        if now_ns() >= limit:
            break
    return mem[OFF_CTL + CTL_BEST_MOVE]
