"""Behavioural tests for the agent: legality, timing, edge cases and known mates.

These run in one process at a fast time control, so they cover far more positions than the
harness can in the same wall time. The harness is still the authority on the protocol and the
clock; this is what finds the rare position that makes the search return something impossible.
"""

import random
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent

# positions that exercise the paths a random game rarely reaches
EDGE_CASES = [
    # promotion available to both sides, en passant on the board
    ("8/PPP2k2/8/8/8/8/2K2ppp/8 w - - 0 1", "promotion race"),
    ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "en passant available"),
    # in check, single reply
    ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", "in check, capture or step aside"),
    # stalemate is one move away for the side to move
    ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "stalemate trap"),
    # bare kings plus a pawn, the endgame scaling path
    ("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", "king and pawn"),
    ("8/8/8/3k4/8/8/8/3K4 w - - 0 1", "bare kings"),
    # castling rights on both sides, both directions
    ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "castling both ways"),
    # a losing position, to make sure a mate against us does not break the search
    ("6k1/5ppp/8/8/8/8/8/1qqqK3 w - - 0 1", "getting mated"),
    # 50 move rule about to trigger
    ("8/8/4k3/8/8/4K3/8/6Q1 w - - 98 120", "halfmove clock at 98"),
]

# every distance below was verified with an independent brute-force mate solver, not by eye
MATES = [
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 1),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 1),
    ("6rk/6pp/8/6N1/8/8/8/6QK w - - 0 1", 1),
    ("6k1/pp4p1/2p5/2bp4/8/P5Pb/1P3rrP/2BRRN1K b - - 0 1", 2),
    ("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1", 2),
    ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", 2),
    ("r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", 3),
    # no mate in sight: the engine must return a sane move without claiming one
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),
    ("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 1", 0),
]


def legal_or_die(board: chess.Board, uci: str, context: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        raise SystemExit(f"{context}: unparsable move {uci!r} at {board.fen()}") from None
    if move not in board.legal_moves:
        raise SystemExit(f"{context}: illegal move {uci} at {board.fen()}")
    return move


def reset() -> None:
    agent.GAME = agent.Game()


def play_game(start: str, budget_ms: int, ply_cap: int) -> tuple[str, int]:
    """Self play from a position, checking every move and every clock."""
    reset()
    board = chess.Board(start)
    clock = {chess.WHITE: float(budget_ms), chess.BLACK: float(budget_ms)}
    plies = 0
    while plies < ply_cap:
        if board.is_game_over(claim_draw=True):
            return board.result(claim_draw=True), plies
        mover = board.turn
        started = time.perf_counter()
        uci = agent.get_move(board.fen(), int(clock[mover]))
        elapsed = (time.perf_counter() - started) * 1000.0
        clock[mover] -= elapsed
        if clock[mover] < 0:
            raise SystemExit(
                f"flagged with {elapsed:.0f}ms on a {clock[mover] + elapsed:.0f}ms clock "
                f"at {board.fen()}"
            )
        clock[mover] += 500
        board.push(legal_or_die(board, uci, "self play"))
        plies += 1
    return "*", plies


def random_openings(count: int, seed: int) -> list[str]:
    """Balanced-ish starting positions, standing in for the platform's unpublished set."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(4, 12)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or not board.legal_moves:
            continue
        out.append(board.fen())
    return out


def check_edges() -> None:
    print("edge cases")
    for fen, label in EDGE_CASES:
        reset()
        board = chess.Board(fen)
        for budget in (200, 3000, 130, 60):
            started = time.perf_counter()
            uci = agent.get_move(board.fen(), budget)
            elapsed = (time.perf_counter() - started) * 1000.0
            legal_or_die(board, uci, label)
            if elapsed > budget + 60:
                raise SystemExit(
                    f"{label}: took {elapsed:.0f}ms of a {budget}ms clock at {board.fen()}"
                )
        print(f"  ok {label}")


def check_mates() -> None:
    print("tactics")
    for fen, expected in MATES:
        reset()
        board = chess.Board(fen)
        uci = agent.get_move(fen, 6000)
        legal_or_die(board, uci, "mate suite")
        score = int(agent.MEM[agent.search.OFF_CTL + agent.search.CTL_BEST_SCORE])
        if expected == 0:
            if score > agent.MATE_IN_MAX:
                raise SystemExit(f"claimed a mate that does not exist at {fen}")
            print(f"  ok no mate available, played {uci} at cp {score}")
            continue
        if score < agent.MATE_IN_MAX:
            raise SystemExit(f"missed mate in {expected} at {fen}: played {uci} score {score}")
        found = (agent.MATE - abs(score) + 1) // 2
        if found > expected:
            raise SystemExit(f"found mate in {found}, wanted {expected}, at {fen}")
        print(f"  ok mate in {found} at {fen.split()[0][:20]}... via {uci}")


def check_repetition() -> None:
    """From a dead-drawn rook ending the engine should not shuffle into a lost tempo, and from
    a winning position it must not repeat. We only assert it never claims a repetition is good
    while it is a piece up."""
    print("repetition awareness")
    reset()
    board = chess.Board("6k1/8/6K1/8/8/8/8/1Q6 w - - 0 1")
    seen: dict[str, int] = {}
    for _ in range(24):
        uci = agent.get_move(board.fen(), 2000)
        board.push(legal_or_die(board, uci, "repetition"))
        key = board.board_fen() + str(board.turn)
        seen[key] = seen.get(key, 0) + 1
        if board.is_game_over(claim_draw=True):
            break
    worst = max(seen.values())
    if worst >= 3:
        raise SystemExit("repeated a position three times while a queen up")
    print(f"  ok, no position seen more than {worst} times while winning")


def check_history() -> None:
    """The tracker must reconstruct the opponent's moves from the FEN alone.

    Repetition scoring is only as good as this: the platform never tells us what the opponent
    played, so the engine replays their reply onto the board it kept. If that ever drifts, the
    search silently loses its view of the game's earlier positions.
    """
    print("game history tracking")
    rng = random.Random(99)
    checked = 0
    plies = 0
    for start in [chess.STARTING_FEN, *random_openings(4, 31337)]:
        reset()
        board = chess.Board(start)
        # the engine's history can only begin at the first position it is actually shown, so
        # when the opening has Black to move the first ply happens before it is ever called
        expected: list[int] = []
        for _ in range(60):
            if board.turn == chess.WHITE:
                if not expected:
                    expected = [agent._key(board)]
                uci = agent.get_move(board.fen(), 400)
                move = legal_or_die(board, uci, "history tracking")
                board.push(move)
                expected.append(agent._key(board))
                checked += 1
                if agent.GAME.keys != expected:
                    raise SystemExit(
                        f"history drifted after {uci} at ply {len(board.move_stack)}: tracker "
                        f"holds {len(agent.GAME.keys)} keys, the game has {len(expected)}"
                    )
            else:
                board.push(rng.choice(list(board.legal_moves)))
                if expected:
                    expected.append(agent._key(board))
            plies += 1
            if board.is_game_over():
                break
    if checked < 30:
        raise SystemExit(f"only {checked} reconstructions checked, the test proved little")
    print(f"  ok, reconstructed the opponent's move {checked} times across {plies} plies")

    # a FEN that drops an en passant square nobody can use must not look like a new game
    reset()
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    agent.get_move(board.fen(), 400)
    before = len(agent.GAME.keys)
    board = chess.Board(agent.GAME.board.fen())  # type: ignore[union-attr]
    double = next(
        move
        for move in board.legal_moves
        if board.piece_type_at(move.from_square) == chess.PAWN
        and abs(move.to_square - move.from_square) == 16
    )
    board.push(double)
    stripped = board.fen().split()
    stripped[3] = "-"
    agent.get_move(" ".join(stripped), 400)
    if len(agent.GAME.keys) != before + 2:
        raise SystemExit(
            f"history reset when the en passant field was dropped: {before} -> "
            f"{len(agent.GAME.keys)} keys"
        )
    print("  ok, an unusable en passant field does not reset the history")


def check_games(games: int, budget_ms: int, seed: int) -> None:
    print(f"self play, {games} games at {budget_ms}ms + 0.5s")
    results: dict[str, int] = {}
    for index, fen in enumerate(random_openings(games, seed)):
        result, plies = play_game(fen, budget_ms, 300)
        results[result] = results.get(result, 0) + 1
        print(f"  game {index + 1}/{games}: {result} in {plies} plies")
    print("  " + ", ".join(f"{key} {value}" for key, value in sorted(results.items())))


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    check_edges()
    check_mates()
    check_repetition()
    check_history()
    check_games(games, budget, 20260903)
    print("\nall engine checks passed")


if __name__ == "__main__":
    main()
