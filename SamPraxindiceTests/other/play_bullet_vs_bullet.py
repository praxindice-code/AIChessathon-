"""Play test games between two of YOUR OWN engine variations (e.g. two different Bullet
checkpoints, or Bullet vs HalfKA) -- each run as its own isolated subprocess via uci_bridge.py,
so they can each load their own weights without colliding.

Clock handling mirrors play_match_elo.py exactly (real per-side clocks, not the "whole
remaining clock passed as a single move's budget" bug from the very first version of this
tool) -- that logic was already validated, this just points both sides at your own engine
instead of Stockfish.

Setup: each variation needs its own complete folder with agent.py, search.py, nnue.py,
board.py, evaluate.py, magics.py, position.py, its own bullet_raw.npz (or weights.qz for
HalfKA), and a copy of uci_bridge.py.

Usage:
    python play_bullet_vs_bullet.py \\
        --engine-a "C:\\...\\variation_A" --label-a "checkpoint-1000" \\
        --engine-b "C:\\...\\variation_B" --label-b "checkpoint-1500" \\
        --games 20 --base 120 --increment 0.5
"""

import argparse
import math
import subprocess
import time

import chess
import chess.pgn


def elo_diff_from_score(score: float, n: int) -> float | None:
    if n == 0:
        return None
    p = score / n
    if p <= 0.0 or p >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / p - 1.0)


def elo_interval(score: float, wins: int, draws: int, losses: int) -> tuple[float, float] | None:
    n = wins + draws + losses
    if n == 0:
        return None
    p = score / n
    if not 0.0 < p < 1.0:
        return None
    spread = (wins * (1 - p) ** 2 + draws * (0.5 - p) ** 2 + losses * p**2) / n
    error = math.sqrt(spread / n)
    band = [
        -400.0 * math.log10(1.0 / value - 1.0)
        for value in (p - 1.96 * error, p + 1.96 * error)
        if 0.0 < value < 1.0
    ]
    return (band[0], band[1]) if len(band) == 2 else None


class EngineProcess:
    """Wraps one engine variation as a subprocess speaking uci_bridge.py's line protocol."""

    def __init__(self, folder: str):
        self.proc = subprocess.Popen(
            ["python", "uci_bridge.py"],
            cwd=folder,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def get_move(self, fen: str, time_left_ms: int) -> str:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(f"{fen}|{time_left_ms}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("engine process closed unexpectedly -- check its stderr")
        return line.strip()

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=5)


def play_game(
    engine_a_folder: str,
    engine_b_folder: str,
    a_is_white: bool,
    base_s: float,
    increment_s: float,
) -> tuple[str, chess.pgn.Game, float, float, int]:
    board = chess.Board()
    engine_a = EngineProcess(engine_a_folder)
    engine_b = EngineProcess(engine_b_folder)
    white_engine, black_engine = (engine_a, engine_b) if a_is_white else (engine_b, engine_a)

    white_clock = base_s
    black_clock = base_s
    a_spent = 0.0
    b_spent = 0.0
    plies = 0

    try:
        while not board.is_game_over(claim_draw=True):
            mover_is_white = board.turn == chess.WHITE
            mover_engine = white_engine if mover_is_white else black_engine
            mover_clock = white_clock if mover_is_white else black_clock

            started = time.perf_counter()
            uci = mover_engine.get_move(board.fen(), int(mover_clock * 1000))
            elapsed = time.perf_counter() - started

            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                result = "0-1" if mover_is_white else "1-0"
                game = chess.pgn.Game.from_board(board)
                game.headers["Result"] = result
                game.headers["Termination"] = f"illegal move {uci}"
                return result, game, a_spent, b_spent, plies

            plies += 1
            if mover_engine is engine_a:
                a_spent += elapsed
            else:
                b_spent += elapsed

            if mover_is_white:
                white_clock += increment_s - elapsed
                if white_clock <= 0.0:
                    board.push(move)
                    game = chess.pgn.Game.from_board(board)
                    game.headers["Result"] = "0-1"
                    game.headers["Termination"] = "white flagged"
                    return "0-1", game, a_spent, b_spent, plies
            else:
                black_clock += increment_s - elapsed
                if black_clock <= 0.0:
                    board.push(move)
                    game = chess.pgn.Game.from_board(board)
                    game.headers["Result"] = "1-0"
                    game.headers["Termination"] = "black flagged"
                    return "1-0", game, a_spent, b_spent, plies

            board.push(move)
    finally:
        engine_a.close()
        engine_b.close()

    game = chess.pgn.Game.from_board(board)
    # Same bug found in play_match_elo.py: board.result() with no arguments does NOT count
    # claimable draws (threefold repetition, 50-move rule), only forced ones. The while-loop
    # above already exits on claim_draw=True, so a claimable-draw ending here would otherwise
    # leave Result as "*", which the caller's scoring logic silently treats as a LOSS.
    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    return result, game, a_spent, b_spent, plies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-a", required=True, help="folder for variation A")
    parser.add_argument("--engine-b", required=True, help="folder for variation B")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base", type=float, default=120.0)
    parser.add_argument("--increment", type=float, default=0.5)
    parser.add_argument("--pgn-out", default="bullet_vs_bullet.pgn")
    args = parser.parse_args()

    a_wins = a_losses = draws = 0
    a_score = 0.0
    a_total = b_total = 0.0
    total_plies = 0

    with open(args.pgn_out, "w") as pgn_file:
        for i in range(args.games):
            a_is_white = i % 2 == 0
            t0 = time.time()
            result, game, a_spent, b_spent, plies = play_game(
                args.engine_a, args.engine_b, a_is_white, args.base, args.increment
            )
            elapsed = time.time() - t0

            game.headers["White"] = args.label_a if a_is_white else args.label_b
            game.headers["Black"] = args.label_b if a_is_white else args.label_a
            print(game, file=pgn_file, end="\n\n")

            if result == "1/2-1/2":
                outcome, delta = "draw", 0.5
            elif (result == "1-0") == a_is_white:
                outcome, delta = f"{args.label_a} win", 1.0
            else:
                outcome, delta = f"{args.label_b} win", 0.0

            a_score += delta
            a_wins += outcome == f"{args.label_a} win"
            a_losses += outcome == f"{args.label_b} win"
            draws += outcome == "draw"
            a_total += a_spent
            b_total += b_spent
            total_plies += plies // 2

            side = "white" if a_is_white else "black"
            print(f"game {i + 1}/{args.games}: {outcome} ({args.label_a} played {side}, {elapsed:.1f}s)")

    n = a_wins + a_losses + draws
    pct = 100 * a_score / n if n else 0.0
    elo = elo_diff_from_score(a_score, n)
    interval = elo_interval(a_score, a_wins, draws, a_losses)

    print(f"\nfinal: {args.label_a} {a_wins}W {a_losses}L {draws}D  ({a_score}/{n} = {pct:.1f}%)")
    if elo is None:
        print("relative Elo: undefined (need a less lopsided result)")
    elif interval is None:
        print(f"relative Elo ({args.label_a} vs {args.label_b}): {elo:+.0f}")
    else:
        print(
            f"relative Elo ({args.label_a} vs {args.label_b}, {args.base:.0f}s+{args.increment}s): "
            f"{elo:+.0f}  (95% interval {interval[0]:+.0f} to {interval[1]:+.0f})"
        )
    if total_plies:
        print(
            f"thinking time: {args.label_a} {a_total / total_plies:.2f}s/move, "
            f"{args.label_b} {b_total / total_plies:.2f}s/move "
            "(these should be close; if not, the Elo is measuring the clock, not strength)"
        )
    print(f"games saved to {args.pgn_out}")


if __name__ == "__main__":
    main()
