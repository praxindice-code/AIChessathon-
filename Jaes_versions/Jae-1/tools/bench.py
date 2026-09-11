"""Fixed-work benchmark: search a set of positions to a fixed depth and report nodes and time.

Depth rather than time keeps the node count identical across runs, so a change that alters the
search tree shows up as a different node count and a change that only alters speed shows up as
a different rate. Both are worth knowing and confusing them wastes hours.
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import search

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N1BP2/PP1QN1PP/R3KB1R b KQ - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "6k1/5ppp/8/3p4/3P4/5N2/5PPP/6K1 w - - 0 1",
    "2r3k1/pp1q1ppp/3bpn2/3p4/3P4/1QN1PN2/PP3PPP/2R3K1 w - - 0 1",
    "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
]


def run(depth: int) -> None:
    total_nodes = 0
    total_time = 0.0
    for fen in POSITIONS:
        agent.GAME = agent.Game()
        board = chess.Board(fen)
        agent._load(board)
        agent._load_history([int(agent.STS[0, 4])])
        agent.TT[:] = 0
        agent.MEM[search.OFF_HISTORY : search.OFF_COUNTER + 8192] = 0
        started = time.perf_counter()
        agent.MEM[search.OFF_CTL + search.CTL_HARD_NS] = int((started + 600.0) * 1e9)
        agent.MEM[search.OFF_CTL + search.CTL_SOFT_NS] = int((started + 600.0) * 1e9)
        search.think(agent.BBS, agent.STS, agent.MBS, agent.MEM, agent.TT, depth)
        elapsed = time.perf_counter() - started
        nodes = int(agent.MEM[search.OFF_CTL + search.CTL_NODES])
        score = int(agent.MEM[search.OFF_CTL + search.CTL_BEST_SCORE])
        total_nodes += nodes
        total_time += elapsed
        print(f"  {nodes:>10,} nodes {elapsed:6.2f}s  cp {score:>6}  {fen}")
    print(
        f"\ntotal {total_nodes:,} nodes in {total_time:.2f}s "
        f"= {total_nodes / total_time / 1e6:.2f} Mnps"
    )


def main() -> None:
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    print(f"benchmark at fixed depth {depth}")
    run(depth)


if __name__ == "__main__":
    main()
