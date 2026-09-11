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
    python play_match_elo.py --stockfish ..\stockfish\stockfish-windows-x86-64-avx2.exe \
        --games 20 --skill 3 --base 60 --increment 0.5
"""

import argparse
import math
import time

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
) -> tuple[str, chess.pgn.Game, float, float, int]:
    board = chess.Board()
    sf = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    agent_clock = base_s
    sf_clock = base_s
    agent_spent = 0.0
    sf_spent = 0.0
    plies = 0
    try:
        sf.configure({"Skill Level": skill})
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
    return game.headers.get("Result", "*"), game, agent_spent, sf_spent, plies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, help="path to stockfish.exe")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--skill", type=int, default=3, help="Stockfish Skill Level, 0-20")
    parser.add_argument("--base", type=float, default=60.0, help="seconds on each clock")
    parser.add_argument("--increment", type=float, default=0.5, help="seconds added per move")
    parser.add_argument("--pgn-out", default="elo_games.pgn")
    args = parser.parse_args()

    wins = losses = draws = 0
    score = 0.0
    agent_total = sf_total = 0.0
    agent_plies = 0

    with open(args.pgn_out, "w") as pgn_file:
        for i in range(args.games):
            agent_is_white = i % 2 == 0
            t0 = time.time()
            result, game, agent_spent, sf_spent, plies = play_game(
                args.stockfish, args.skill, agent_is_white, args.base, args.increment
            )
            elapsed = time.time() - t0

            opponent = f"Stockfish(skill={args.skill})"
            game.headers["White"] = "agent.py" if agent_is_white else opponent
            game.headers["Black"] = opponent if agent_is_white else "agent.py"
            print(game, file=pgn_file, end="\n\n")

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
            agent_total += agent_spent
            sf_total += sf_spent
            agent_plies += plies // 2

            side = "white" if agent_is_white else "black"
            per_move = agent_spent / max(1, plies // 2)
            print(
                f"game {i + 1}/{args.games}: {outcome} (agent played {side}, {elapsed:.1f}s, "
                f"agent {per_move:.2f}s/move)"
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
            "(these should be close; if they are not, the Elo is measuring the clock)"
        )
    print(f"games saved to {args.pgn_out}")


if __name__ == "__main__":
    main()
