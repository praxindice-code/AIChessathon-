"""Play test games between our engine (imported in-process) and Stockfish, and report Elo.

Same idea as the original, with one correction that changes the answer completely.

The original passed `--time` straight into `agent.get_move(fen, time_left_ms)`. But that
argument is the whole remaining clock, not a per-move allowance: the engine divides it by the
moves it expects to still play. Handing it 1000 buys about 18ms of thinking while Stockfish
gets a full second on the same move -- a fifty-fold handicap, and an Elo number measured
against it means nothing.

So each side now gets a real clock. Our agent is told what is genuinely left on its own clock,
exactly as the competition does it, and Stockfish is given the same base and increment through
its own clock control. Both sides also have their actual thinking time measured and printed,
so parity is something you can see in the output rather than something you have to trust.

Usage:
    python play_match_elo.py --stockfish ..\\stockfish\\stockfish-windows-x86-64-avx2.exe \\
        --games 100 --workers 3 --skill 3 --sf-threads 1 --sf-hash 64 --base 60 --increment 0.5
"""

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import chess.engine
import chess.pgn

import agent


def elo_diff_from_score(score: float, n: int) -> float | None:
    """Logistic Elo-difference estimate from a score fraction, or None at the boundaries."""
    if n == 0:
        return None
    p = score / n
    if p <= 0.0 or p >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / p - 1.0)


def elo_interval(score: float, wins: int, draws: int, losses: int) -> tuple[float, float] | None:
    """95% interval on the Elo estimate, so a 20-game result is not read as precise."""
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


def play_game(
    stockfish_path: str,
    skill: int,
    agent_is_white: bool,
    base_s: float,
    increment_s: float,
    sf_threads: int = 1,
    sf_hash: int = 64,
) -> tuple[str, chess.pgn.Game, float, float, int]:
    board = chess.Board()
    sf = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    agent_clock = base_s
    sf_clock = base_s
    agent_spent = 0.0
    sf_spent = 0.0
    plies = 0
    try:
        sf.configure({
            "Skill Level": skill,
            "Threads": max(1, sf_threads),
            "Hash": max(16, sf_hash),
        })
        while not board.is_game_over(claim_draw=True):
            mover_is_agent = (board.turn == chess.WHITE) == agent_is_white
            started = time.perf_counter()
            if mover_is_agent:
                # the whole remaining clock, in milliseconds: what the competition passes
                uci = agent.get_move(board.fen(), int(agent_clock * 1000))
                move = chess.Move.from_uci(uci)
            else:
                white_clock = sf_clock if not agent_is_white else agent_clock
                black_clock = sf_clock if agent_is_white else agent_clock
                result = sf.play(
                    board,
                    chess.engine.Limit(
                        white_clock=white_clock,
                        black_clock=black_clock,
                        white_inc=increment_s,
                        black_inc=increment_s,
                    ),
                )
                move = result.move
                if move is None:
                    break
            elapsed = time.perf_counter() - started
            plies += 1
            if mover_is_agent:
                agent_spent += elapsed
                agent_clock += increment_s - elapsed
                if agent_clock <= 0.0:
                    # flagging is a real loss in the competition, so it is a real loss here
                    board.push(move)
                    game = chess.pgn.Game.from_board(board)
                    game.headers["Result"] = "0-1" if agent_is_white else "1-0"
                    game.headers["Termination"] = "agent flagged"
                    return game.headers["Result"], game, agent_spent, sf_spent, plies
            else:
                sf_spent += elapsed
                sf_clock += increment_s - elapsed
            board.push(move)
    finally:
        sf.quit()

    game = chess.pgn.Game.from_board(board)
    # board.result() with no arguments does NOT count claimable draws (threefold repetition,
    # 50-move rule) as game-over -- only forced ones (checkmate, stalemate, insufficient
    # material). The while-loop above already exits on claim_draw=True, so a claimable-draw
    # ending here would otherwise leave Result as "*", which the caller's scoring logic
    # silently treats as a LOSS, not a draw. Confirmed as a real bug: several games in a real
    # run ended via genuine threefold repetition and were miscounted as losses.
    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    return result, game, agent_spent, sf_spent, plies



def _play_game_worker(job):
    """Top-level helper so Windows ProcessPoolExecutor can pickle it."""
    (
        game_index,
        stockfish_path,
        skill,
        agent_is_white,
        base_s,
        increment_s,
        sf_threads,
        sf_hash,
    ) = job

    t0 = time.time()
    result, game, agent_spent, sf_spent, plies = play_game(
        stockfish_path,
        skill,
        agent_is_white,
        base_s,
        increment_s,
        sf_threads,
        sf_hash,
    )
    elapsed = time.time() - t0

    opponent = f"Stockfish(skill={skill})"
    game.headers["White"] = "agent.py" if agent_is_white else opponent
    game.headers["Black"] = opponent if agent_is_white else "agent.py"

    # Never return chess.pgn.Game through ProcessPoolExecutor.
    # Long games create a deep parent/child object graph which can exceed
    # Python's recursion limit while multiprocessing tries to pickle it.
    pgn_text = str(game)

    return {
        "index": game_index,
        "result": result,
        "pgn": pgn_text,
        "agent_spent": agent_spent,
        "sf_spent": sf_spent,
        "plies": plies,
        "elapsed": elapsed,
        "agent_is_white": agent_is_white,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, help="path to stockfish.exe")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--skill", type=int, default=3, help="Stockfish Skill Level, 0-20")
    parser.add_argument("--sf-threads", type=int, default=1,
                        help="Stockfish threads per game; keep at 1 when running games in parallel")
    parser.add_argument("--sf-hash", type=int, default=64,
                        help="Stockfish hash MB per game; 64 MB is enough for short benchmark games")
    parser.add_argument("--base", type=float, default=60.0, help="seconds on each clock")
    parser.add_argument("--increment", type=float, default=0.5, help="seconds added per move")
    parser.add_argument("--pgn-out", default="elo_games.pgn")
    parser.add_argument(
        "--workers", type=int, default=3,
        help="concurrent games; default 3 is tuned for Ryzen 5 5600G + RTX 3060"
    )
    args = parser.parse_args()

    workers = max(1, min(args.workers, args.games))
    wins = losses = draws = 0
    score = 0.0
    agent_total = sf_total = 0.0
    agent_plies = 0

    jobs = [
        (
            i,
            args.stockfish,
            args.skill,
            i % 2 == 0,
            args.base,
            args.increment,
            args.sf_threads,
            args.sf_hash,
        )
        for i in range(args.games)
    ]

    print(
        f"running {args.games} games with {workers} concurrent worker(s) "
        f"(PID {os.getpid()}), Stockfish threads/game={args.sf_threads}, "
        f"hash/game={args.sf_hash} MB"
    )
    if workers > 1:
        print(
            "NOTE: each worker is a separate Python process. If agent.py loads the NN on CUDA, "
            "each worker may allocate its own GPU model/VRAM. On an RTX 3060, start at 3 workers; "
"try 4 only if VRAM and GPU utilization have headroom."
        )

    # Persist each completed PGN immediately. If a later game fails or the
    # process is interrupted, already-finished games remain on disk.
    with open(args.pgn_out, "w", encoding="utf-8", buffering=1) as pgn_file:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_play_game_worker, job) for job in jobs]

            for future in as_completed(futures):
                try:
                    item = future.result()
                except Exception as exc:
                    # One failed game should not destroy the rest of the batch.
                    print(
                        f"worker game failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                # Save first, then update stats.
                pgn_file.write(item["pgn"])
                pgn_file.write("\n\n")
                pgn_file.flush()

                result = item["result"]
                agent_is_white = item["agent_is_white"]

                if result == "1/2-1/2":
                    outcome, delta = "draw", 0.5
                elif (result == "1-0") == agent_is_white:
                    outcome, delta = "win", 1.0
                else:
                    outcome, delta = "loss", 0.0

                score += delta
                wins += outcome == "win"
                losses += outcome == "loss"
                draws += outcome == "draw"
                agent_total += item["agent_spent"]
                sf_total += item["sf_spent"]
                agent_plies += item["plies"] // 2

                side = "white" if agent_is_white else "black"
                per_move = item["agent_spent"] / max(1, item["plies"] // 2)
                print(
                    f"game {item['index'] + 1}/{args.games}: {outcome} "
                    f"(agent played {side}, {item['elapsed']:.1f}s, "
                    f"agent {per_move:.2f}s/move)",
                    flush=True,
                )

    n = wins + losses + draws
    pct = 100 * score / n if n else 0.0
    elo = elo_diff_from_score(score, n)
    interval = elo_interval(score, wins, draws, losses)

    print(f"\nfinal: {wins}W {losses}L {draws}D  ({score}/{n} = {pct:.1f}%)")
    if elo is None:
        print("relative Elo: undefined (need a less lopsided result)")
    elif interval is None:
        print(f"relative Elo vs Stockfish(skill={args.skill}): {elo:+.0f}")
    else:
        print(
            f"relative Elo vs Stockfish(skill={args.skill}, {args.base:.0f}s+{args.increment}s): "
            f"{elo:+.0f}  (95% interval {interval[0]:+.0f} to {interval[1]:+.0f})"
        )
    if agent_plies:
        print(
            f"thinking time: agent {agent_total / agent_plies:.2f}s/move, "
            f"stockfish {sf_total / max(1, agent_plies):.2f}s/move "
            "(with parallel workers, wall-clock contention can change these numbers)"
        )
    print(f"games saved to {args.pgn_out}")

if __name__ == "__main__":
    main()
