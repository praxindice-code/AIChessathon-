"""Prove the jitted move generator against python-chess.

Perft counts every legal leaf at a depth. If the generator disagrees with python-chess by a
single node, the engine will eventually emit an illegal move and lose a game on the spot, so
this runs both the published perft numbers and a random-game cross-check against python-chess.
"""

import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board import gen_moves, in_check, make_move, perft, zobrist
from position import from_board, to_uci

# position, depth, expected leaves; the standard suite plus a promotion-heavy endgame
CASES = [
    (chess.STARTING_FEN, 5, 4865609),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594),
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 6, 72339),
]

MAXPLY = 24


def stacks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((MAXPLY, 8), dtype=np.int64),
        np.zeros((MAXPLY, 6), dtype=np.int64),
        np.zeros((MAXPLY, 64), dtype=np.int8),
        np.zeros(MAXPLY * 320, dtype=np.int32),
    )


def run_perft(fen: str, depth: int) -> tuple[int, float]:
    bbs, sts, mbs, moves = stacks()
    bb, st, mb = from_board(chess.Board(fen))
    bbs[0] = bb
    sts[0] = st
    mbs[0] = mb
    sts[0, 4] = zobrist(bb, st)
    started = time.perf_counter()
    nodes = perft(bbs, sts, mbs, moves, 0, depth)
    return nodes, time.perf_counter() - started


def legal_set(board: chess.Board) -> set[str]:
    bbs, sts, mbs, moves = stacks()
    bb, st, mb = from_board(board)
    bbs[0] = bb
    sts[0] = st
    mbs[0] = mb
    end = gen_moves(bbs[0], sts[0], mbs[0], moves, 0)
    out = set()
    for index in range(end):
        make_move(bbs, sts, mbs, 0, moves[index])
        if not in_check(bbs[1], st[0]):
            out.add(to_uci(int(moves[index])))
    return out


def cross_check(games: int, seed: int) -> None:
    """Play random games and demand the same legal move set as python-chess at every ply."""
    rng = random.Random(seed)
    for game in range(games):
        board = chess.Board()
        while not board.is_game_over(claim_draw=False) and board.fullmove_number < 120:
            expected = {move.uci() for move in board.legal_moves}
            got = legal_set(board)
            if expected != got:
                raise SystemExit(
                    f"generator disagrees at {board.fen()}\n"
                    f"  missing {sorted(expected - got)}\n  extra {sorted(got - expected)}"
                )
            board.push(rng.choice(list(board.legal_moves)))
        if (game + 1) % 25 == 0:
            print(f"  cross-checked {game + 1} random games")


def hash_check(games: int, seed: int) -> None:
    """The incremental zobrist must always equal a hash computed from scratch."""
    rng = random.Random(seed)
    bbs, sts, mbs, moves = stacks()
    for _ in range(games):
        board = chess.Board()
        bb, st, mb = from_board(board)
        bbs[0] = bb
        sts[0] = st
        mbs[0] = mb
        sts[0, 4] = zobrist(bb, st)
        while not board.is_game_over() and board.fullmove_number < 100:
            end = gen_moves(bbs[0], sts[0], mbs[0], moves, 0)
            legal = []
            for index in range(end):
                make_move(bbs, sts, mbs, 0, moves[index])
                if not in_check(bbs[1], sts[0, 0]):
                    legal.append(int(moves[index]))
            if not legal:
                break
            chosen = rng.choice(legal)
            make_move(bbs, sts, mbs, 0, chosen)
            fresh = zobrist(bbs[1], sts[1])
            if fresh != sts[1, 4]:
                raise SystemExit(f"zobrist drift after {to_uci(chosen)} from {board.fen()}")
            board.push(chess.Move.from_uci(to_uci(chosen)))
            bbs[0] = bbs[1]
            sts[0] = sts[1]
            mbs[0] = mbs[1]


def main() -> None:
    print("warming the jit")
    started = time.perf_counter()
    run_perft(chess.STARTING_FEN, 1)
    print(f"  compiled in {time.perf_counter() - started:.1f}s")

    failures = 0
    total_nodes = 0
    total_time = 0.0
    for fen, depth, expected in CASES:
        nodes, elapsed = run_perft(fen, depth)
        total_nodes += nodes
        total_time += elapsed
        status = "ok " if nodes == expected else "FAIL"
        failures += nodes != expected
        print(
            f"  {status} depth {depth} {nodes:>10,} (want {expected:>10,})"
            f"  {elapsed:6.2f}s  {fen}"
        )
    print(f"\nperft total {total_nodes:,} nodes in {total_time:.2f}s "
          f"({total_nodes / max(total_time, 1e-9) / 1e6:.2f} Mnps)")

    print("\ncross-checking the generator against python-chess on random games")
    cross_check(60, 7)
    print("cross-checking incremental zobrist")
    hash_check(40, 11)
    print("hash ok")

    if failures:
        raise SystemExit(f"{failures} perft mismatches")
    print("\nall move generation checks passed")


if __name__ == "__main__":
    main()
