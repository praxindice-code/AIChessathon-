"""Symmetry checks on the evaluation.

A hand-written evaluation is easy to get subtly wrong in one colour only: a mirrored index, a
sign flipped in one branch, a table written for White and reused for Black. Those bugs cost
half a pawn in one direction and nothing in the other, which is invisible in a game log and
obvious here. Mirroring a position vertically and swapping colours must negate the score, up
to the tempo bonus that is deliberately asymmetric.
"""

import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board import zobrist
from evaluate import TEMPO, evaluate
from position import from_board


def mirror(board: chess.Board) -> chess.Board:
    """Flip the board vertically and swap colours, keeping castling rights consistent."""
    flipped = chess.Board(None)
    for square, piece in board.piece_map().items():
        flipped.set_piece_at(
            chess.square(chess.square_file(square), 7 - chess.square_rank(square)),
            chess.Piece(piece.piece_type, not piece.color),
        )
    flipped.turn = not board.turn
    rights = ""
    if board.has_kingside_castling_rights(chess.BLACK):
        rights += "K"
    if board.has_queenside_castling_rights(chess.BLACK):
        rights += "Q"
    if board.has_kingside_castling_rights(chess.WHITE):
        rights += "k"
    if board.has_queenside_castling_rights(chess.WHITE):
        rights += "q"
    flipped.set_castling_fen(rights if rights else "-")
    if board.ep_square is not None:
        flipped.ep_square = chess.square(
            chess.square_file(board.ep_square), 7 - chess.square_rank(board.ep_square)
        )
    return flipped


def score(board: chess.Board) -> int:
    bb, st, _ = from_board(board)
    st[4] = zobrist(bb, st)
    return int(evaluate(bb, st))


def positions(count: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(2, 70)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if board.is_game_over() or board.is_check():
            continue
        out.append(board)
    return out


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    worst = 0
    worst_fen = ""
    boards = positions(count, 4242)
    for board in boards:
        direct = score(board)
        flipped = score(mirror(board))
        # both scores are from the side to move, and both sides move in their own position,
        # so a symmetric evaluation returns exactly the same number
        gap = abs(direct - flipped)
        if gap > worst:
            worst = gap
            worst_fen = board.fen()
    print(f"checked {len(boards)} positions, worst mirror gap {worst} cp")
    if worst_fen:
        print(f"  worst at {worst_fen}")
    if worst != 0:
        raise SystemExit("the evaluation is not colour symmetric")

    start = score(chess.Board())
    print(f"start position scores {start} cp for the side to move (tempo is {TEMPO})")
    if start != TEMPO:
        raise SystemExit(f"the opening position should score exactly the tempo bonus, got {start}")
    print("evaluation symmetry ok")


if __name__ == "__main__":
    main()
