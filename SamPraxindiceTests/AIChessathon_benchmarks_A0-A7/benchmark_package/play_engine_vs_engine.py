# Robust head-to-head benchmark for two agent.py chess engine folders.
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess

BRIDGE_CODE = r'''
import sys
import agent
print("__READY__", flush=True)
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    if raw == "__QUIT__":
        break
    try:
        fen, ms = raw.rsplit("|", 1)
        move = agent.get_move(fen, int(ms))
        print(move, flush=True)
    except BaseException as exc:
        print("__ERROR__:" + repr(exc), flush=True)
'''

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_PROCS: set[subprocess.Popen] = set()
_STOP = threading.Event()


def _register(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS.add(proc)


def _unregister(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS.discard(proc)


def terminate_all_active_engines() -> None:
    _STOP.set()
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE_PROCS)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    deadline = time.monotonic() + 1.5
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


class EngineProcess:
    def __init__(self, folder: str, label: str):
        self.folder = str(Path(folder).resolve())
        self.label = label
        if not Path(self.folder, "agent.py").is_file():
            raise FileNotFoundError(f"{label}: no agent.py in {self.folder}")

        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", BRIDGE_CODE],
            cwd=self.folder,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        _register(self.proc)

        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline()
        if ready.strip() != "__READY__":
            rc = self.proc.poll()
            self.close(force=True)
            raise RuntimeError(
                f"{label} failed during startup (rc={rc}); expected READY, got {ready!r}. "
                "Check the engine stderr above."
            )

    def get_move(self, fen: str, time_left_ms: int) -> str:
        if _STOP.is_set():
            raise InterruptedError("benchmark stopping")
        if self.proc.poll() is not None:
            raise RuntimeError(f"{self.label} process exited with code {self.proc.returncode}")
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(f"{fen}|{max(0, int(time_left_ms))}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.label} process closed unexpectedly")
        line = line.strip()
        if line.startswith("__ERROR__:"):
            raise RuntimeError(f"{self.label} bridge error: {line[10:]}")
        return line

    def close(self, force: bool = False) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None and not force and proc.stdin is not None:
                try:
                    proc.stdin.write("__QUIT__\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.poll() is None:
                try:
                    proc.wait(timeout=2.0 if not force else 0.2)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
        finally:
            _unregister(proc)


@dataclass
class GameResult:
    index: int
    a_is_white: bool
    result: str
    termination: str
    start_fen: str
    moves: list[str]
    a_spent: float
    b_spent: float
    a_moves: int
    b_moves: int
    elapsed_wall: float


def _result_for_forfeit(loser_is_white: bool) -> str:
    return "0-1" if loser_is_white else "1-0"


def _natural_termination(board: chess.Board) -> str:
    # Compatibility-safe termination labeling for older python-chess versions.
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"

    can_claim_fifty = getattr(board, "can_claim_fifty_moves", None)
    if can_claim_fifty is not None and can_claim_fifty():
        return "fifty-move rule"

    can_claim_threefold = getattr(board, "can_claim_threefold_repetition", None)
    if can_claim_threefold is not None and can_claim_threefold():
        return "threefold repetition"

    is_seventyfive = getattr(board, "is_seventyfive_moves", None)
    if is_seventyfive is not None and is_seventyfive():
        return "seventyfive-move rule"

    is_fivefold = getattr(board, "is_fivefold_repetition", None)
    if is_fivefold is not None and is_fivefold():
        return "fivefold repetition"

    return "draw"


def play_game(
    index: int,
    engine_a_folder: str,
    engine_b_folder: str,
    label_a: str,
    label_b: str,
    a_is_white: bool,
    base_s: float,
    increment_s: float,
    start_fen: str,
) -> GameResult:
    wall0 = time.perf_counter()
    board = chess.Board(start_fen)
    moves: list[str] = []
    engine_a: Optional[EngineProcess] = None
    engine_b: Optional[EngineProcess] = None
    a_spent = b_spent = 0.0
    a_moves = b_moves = 0

    try:
        engine_a = EngineProcess(engine_a_folder, label_a)
        engine_b = EngineProcess(engine_b_folder, label_b)
        white_engine, black_engine = (engine_a, engine_b) if a_is_white else (engine_b, engine_a)

        white_clock = float(base_s)
        black_clock = float(base_s)

        while not board.is_game_over(claim_draw=True):
            if _STOP.is_set():
                raise InterruptedError("benchmark stopping")

            mover_is_white = board.turn == chess.WHITE
            mover_engine = white_engine if mover_is_white else black_engine
            mover_clock = white_clock if mover_is_white else black_clock

            started = time.perf_counter()
            try:
                uci = mover_engine.get_move(board.fen(), int(mover_clock * 1000.0))
            except InterruptedError:
                raise
            except Exception as exc:
                result = _result_for_forfeit(mover_is_white)
                return GameResult(
                    index, a_is_white, result,
                    f"{mover_engine.label} engine error: {exc}",
                    start_fen, moves, a_spent, b_spent, a_moves, b_moves,
                    time.perf_counter() - wall0,
                )
            elapsed = time.perf_counter() - started

            if mover_engine is engine_a:
                a_spent += elapsed
                a_moves += 1
            else:
                b_spent += elapsed
                b_moves += 1

            # Correct clock rule: flag BEFORE increment is awarded.
            remaining_after_move = mover_clock - elapsed
            if remaining_after_move < 0.0:
                result = _result_for_forfeit(mover_is_white)
                return GameResult(
                    index, a_is_white, result,
                    f"{mover_engine.label} flagged",
                    start_fen, moves, a_spent, b_spent, a_moves, b_moves,
                    time.perf_counter() - wall0,
                )

            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                result = _result_for_forfeit(mover_is_white)
                return GameResult(
                    index, a_is_white, result,
                    f"{mover_engine.label} malformed move {uci!r}",
                    start_fen, moves, a_spent, b_spent, a_moves, b_moves,
                    time.perf_counter() - wall0,
                )

            if move not in board.legal_moves:
                result = _result_for_forfeit(mover_is_white)
                return GameResult(
                    index, a_is_white, result,
                    f"{mover_engine.label} illegal move {uci}",
                    start_fen, moves, a_spent, b_spent, a_moves, b_moves,
                    time.perf_counter() - wall0,
                )

            moves.append(uci)
            board.push(move)
            if mover_is_white:
                white_clock = remaining_after_move + increment_s
            else:
                black_clock = remaining_after_move + increment_s

        result = board.result(claim_draw=True)
        return GameResult(
            index, a_is_white, result, _natural_termination(board),
            start_fen, moves, a_spent, b_spent, a_moves, b_moves,
            time.perf_counter() - wall0,
        )
    finally:
        if engine_a is not None:
            engine_a.close(force=_STOP.is_set())
        if engine_b is not None:
            engine_b.close(force=_STOP.is_set())


def elo_diff_from_score(score: float, n: int) -> Optional[float]:
    if n <= 0:
        return None
    p = score / n
    if not 0.0 < p < 1.0:
        return None
    return 400.0 * math.log10(p / (1.0 - p))


def elo_interval(score: float, wins: int, draws: int, losses: int) -> Optional[tuple[float, float]]:
    n = wins + draws + losses
    if n <= 0:
        return None
    p = score / n
    if not 0.0 < p < 1.0:
        return None
    variance = (wins * (1.0 - p) ** 2 + draws * (0.5 - p) ** 2 + losses * p**2) / n
    se = math.sqrt(variance / n)
    lo_p = max(1e-9, p - 1.96 * se)
    hi_p = min(1.0 - 1e-9, p + 1.96 * se)
    return (
        400.0 * math.log10(lo_p / (1.0 - lo_p)),
        400.0 * math.log10(hi_p / (1.0 - hi_p)),
    )


def _escape_tag(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_pgn_text(gr: GameResult, label_a: str, label_b: str) -> str:
    white = label_a if gr.a_is_white else label_b
    black = label_b if gr.a_is_white else label_a
    tags = [
        ("Event", "Engine head-to-head benchmark"),
        ("Site", "?"),
        ("Date", time.strftime("%Y.%m.%d")),
        ("Round", str(gr.index + 1)),
        ("White", white),
        ("Black", black),
        ("Result", gr.result),
        ("Termination", gr.termination),
    ]
    if gr.start_fen != chess.STARTING_FEN:
        tags.append(("SetUp", "1"))
        tags.append(("FEN", gr.start_fen))

    out = [f'[{k} "{_escape_tag(v)}"]' for k, v in tags]
    out.append("")

    board = chess.Board(gr.start_fen)
    tokens: list[str] = []
    for uci in gr.moves:
        move = chess.Move.from_uci(uci)
        if board.turn == chess.WHITE:
            tokens.append(f"{board.fullmove_number}.")
        elif not tokens:
            tokens.append(f"{board.fullmove_number}...")
        tokens.append(board.san(move))
        board.push(move)
    tokens.append(gr.result)

    line = ""
    for tok in tokens:
        candidate = tok if not line else line + " " + tok
        if len(candidate) > 100 and line:
            out.append(line)
            line = tok
        else:
            line = candidate
    if line:
        out.append(line)
    return "\n".join(out) + "\n\n"


def load_openings(path: Optional[str]) -> list[str]:
    if not path:
        return [chess.STARTING_FEN]
    fens: list[str] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            board = chess.Board(line)
        except ValueError as exc:
            raise ValueError(f"bad FEN on {path}:{lineno}: {exc}") from exc
        fens.append(board.fen())
    if not fens:
        raise ValueError(f"no openings found in {path}")
    return fens


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark two agent.py chess engines head-to-head")
    parser.add_argument("--engine-a", required=True, help="folder containing engine A's agent.py")
    parser.add_argument("--engine-b", required=True, help="folder containing engine B's agent.py")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1, help="concurrent games; each game runs two engine processes")
    parser.add_argument("--base", type=float, default=60.0)
    parser.add_argument("--increment", type=float, default=0.5)
    parser.add_argument("--pgn-out", default="engine_vs_engine.pgn")
    parser.add_argument("--openings", default=None, help="optional file with one FEN per line; colors are paired")
    args = parser.parse_args()

    if args.games <= 0 or args.workers <= 0:
        parser.error("--games and --workers must be positive")
    if args.base <= 0 or args.increment < 0:
        parser.error("--base must be > 0 and --increment must be >= 0")

    engine_a = str(Path(args.engine_a).resolve())
    engine_b = str(Path(args.engine_b).resolve())
    if engine_a == engine_b:
        print("WARNING: --engine-a and --engine-b resolve to the same folder.", file=sys.stderr)
    for folder, label in ((engine_a, args.label_a), (engine_b, args.label_b)):
        if not Path(folder, "agent.py").is_file():
            parser.error(f"{label}: no agent.py found in {folder}")

    openings = load_openings(args.openings)
    print(
        f"running {args.games} games, {args.workers} concurrent game(s), "
        f"{args.base:g}+{args.increment:g}; {args.label_a} vs {args.label_b}"
    )
    if args.workers > 1:
        print(f"NOTE: {args.workers} games = up to {args.workers * 2} engine processes at once.")

    wins = losses = draws = 0
    score = 0.0
    a_time = b_time = 0.0
    a_moves = b_moves = 0
    completed = 0

    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="chess-game")
    futures = []
    try:
        for i in range(args.games):
            a_is_white = (i % 2 == 0)
            opening = openings[(i // 2) % len(openings)]
            futures.append(executor.submit(
                play_game,
                i, engine_a, engine_b, args.label_a, args.label_b,
                a_is_white, args.base, args.increment, opening,
            ))

        with open(args.pgn_out, "w", encoding="utf-8", buffering=1) as pgn_file:
            for fut in as_completed(futures):
                try:
                    gr = fut.result()
                except CancelledError:
                    continue
                except InterruptedError:
                    continue
                except Exception as exc:
                    print(f"worker/game failure: {exc!r}", file=sys.stderr, flush=True)
                    continue

                pgn_file.write(build_pgn_text(gr, args.label_a, args.label_b))
                pgn_file.flush()
                completed += 1

                if gr.result == "1/2-1/2":
                    draws += 1
                    score += 0.5
                    outcome = "draw"
                else:
                    a_won = (gr.result == "1-0") == gr.a_is_white
                    if a_won:
                        wins += 1
                        score += 1.0
                        outcome = f"{args.label_a} win"
                    else:
                        losses += 1
                        outcome = f"{args.label_b} win"

                a_time += gr.a_spent
                b_time += gr.b_spent
                a_moves += gr.a_moves
                b_moves += gr.b_moves
                side = "white" if gr.a_is_white else "black"
                print(
                    f"game {gr.index + 1}/{args.games}: {outcome} "
                    f"({args.label_a} {side}, {gr.elapsed_wall:.1f}s, {gr.termination})",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nCtrl+C received: preserving completed PGNs and stopping active engines...", flush=True)
        _STOP.set()
        for fut in futures:
            fut.cancel()
        terminate_all_active_engines()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    n = wins + losses + draws
    pct = 100.0 * score / n if n else 0.0
    print(f"\n{'partial' if completed < args.games else 'final'}: {args.label_a} {wins}W {losses}L {draws}D ({score:g}/{n} = {pct:.1f}%)")
    elo = elo_diff_from_score(score, n)
    ci = elo_interval(score, wins, draws, losses)
    if elo is not None:
        if ci is None:
            print(f"relative Elo ({args.label_a} vs {args.label_b}): {elo:+.0f}")
        else:
            print(f"relative Elo ({args.label_a} vs {args.label_b}): {elo:+.0f} (95% interval {ci[0]:+.0f} to {ci[1]:+.0f})")
    else:
        print("relative Elo: undefined for a 0%/100% score")
    if a_moves:
        print(f"{args.label_a} mean think time: {a_time / a_moves:.3f}s/move")
    if b_moves:
        print(f"{args.label_b} mean think time: {b_time / b_moves:.3f}s/move")
    print(f"completed PGNs saved to {args.pgn_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
