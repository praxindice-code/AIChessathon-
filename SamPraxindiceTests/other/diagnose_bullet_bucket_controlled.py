import chess
import board as board_mod
import position
import nnue as bn


def eval_board(board):
    bb, st, mb = position.from_board(board)
    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)
    side = st[0]
    piece_count = board_mod.popcount(bb[0] | bb[1])
    return bn.nnue_eval(acc, psqt, side, piece_count, 0), piece_count


def full_material_board(turn):
    """Standard starting material, but as a bare piece placement (no castling rights/etc
    complications), so we can cleanly remove/add exactly one piece per test."""
    board = chess.Board(None)
    back_rank = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN,
                 chess.KING, chess.BISHOP, chess.KNIGHT, chess.ROOK]
    for file in range(8):
        board.set_piece_at(chess.square(file, 0), chess.Piece(back_rank[file], chess.WHITE))
        board.set_piece_at(chess.square(file, 1), chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.square(file, 6), chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.square(file, 7), chess.Piece(back_rank[file], chess.BLACK))
    board.turn = turn
    board.castling_rights = 0
    board.ep_square = None
    return board


def test_piece_value(label, turn, colour, piece_type, remove_square):
    """Compares eval WITH vs WITHOUT one piece, holding total piece count at 31-32 throughout
    so the output bucket (bucket 7 for both) never changes across the comparison."""
    with_piece = full_material_board(turn)
    (eval_with, pieces_with) = eval_board(with_piece)

    without_piece = with_piece.copy()
    without_piece.remove_piece_at(remove_square)
    (eval_without, pieces_without) = eval_board(without_piece)

    delta = eval_with - eval_without
    print(
        f"{label:<40} pieces={pieces_with}/{pieces_without}  "
        f"with={eval_with:>9.4f}  without={eval_without:>9.4f}  delta={delta:>+9.4f}"
    )


print()
print("=" * 90)
print("BUCKET-CONTROLLED MATERIAL TEST -- every comparison stays in the same output bucket")
print("(31-32 total pieces throughout, matching the bucket the queen-ablation test used)")
print("=" * 90)

print()
print("WHITE TO MOVE -- white's own pieces (delta should be POSITIVE: having them helps the mover)")
print("-" * 90)
test_piece_value("White queen (d1)",  chess.WHITE, chess.WHITE, chess.QUEEN,  chess.D1)
test_piece_value("White rook (a1)",   chess.WHITE, chess.WHITE, chess.ROOK,   chess.A1)
test_piece_value("White bishop (c1)", chess.WHITE, chess.WHITE, chess.BISHOP, chess.C1)
test_piece_value("White knight (b1)", chess.WHITE, chess.WHITE, chess.KNIGHT, chess.B1)
test_piece_value("White pawn (a2)",   chess.WHITE, chess.WHITE, chess.PAWN,   chess.A2)

print()
print("WHITE TO MOVE -- black's pieces (delta should be NEGATIVE: opponent having them hurts the mover)")
print("-" * 90)
test_piece_value("Black queen (d8)",  chess.WHITE, chess.BLACK, chess.QUEEN,  chess.D8)
test_piece_value("Black rook (a8)",   chess.WHITE, chess.BLACK, chess.ROOK,   chess.A8)
test_piece_value("Black bishop (c8)", chess.WHITE, chess.BLACK, chess.BISHOP, chess.C8)
test_piece_value("Black knight (b8)", chess.WHITE, chess.BLACK, chess.KNIGHT, chess.B8)
test_piece_value("Black pawn (a7)",   chess.WHITE, chess.BLACK, chess.PAWN,   chess.A7)

print()
print("BLACK TO MOVE -- black's own pieces (delta should be POSITIVE)")
print("-" * 90)
test_piece_value("Black queen (d8)",  chess.BLACK, chess.BLACK, chess.QUEEN,  chess.D8)
test_piece_value("Black rook (a8)",   chess.BLACK, chess.BLACK, chess.ROOK,   chess.A8)
test_piece_value("Black bishop (c8)", chess.BLACK, chess.BLACK, chess.BISHOP, chess.C8)
test_piece_value("Black knight (b8)", chess.BLACK, chess.BLACK, chess.KNIGHT, chess.B8)
test_piece_value("Black pawn (a7)",   chess.BLACK, chess.BLACK, chess.PAWN,   chess.A7)

print()
print("BLACK TO MOVE -- white's pieces (delta should be NEGATIVE)")
print("-" * 90)
test_piece_value("White queen (d1)",  chess.BLACK, chess.WHITE, chess.QUEEN,  chess.D1)
test_piece_value("White rook (a1)",   chess.BLACK, chess.WHITE, chess.ROOK,   chess.A1)
test_piece_value("White bishop (c1)", chess.BLACK, chess.WHITE, chess.BISHOP, chess.C1)
test_piece_value("White knight (b1)", chess.BLACK, chess.WHITE, chess.KNIGHT, chess.B1)
test_piece_value("White pawn (a2)",   chess.BLACK, chess.WHITE, chess.PAWN,   chess.A2)

print()
print("=" * 90)
print("Expected: queen > rook > bishop/knight > pawn in |delta|, with the correct sign per")
print("section above. If this now looks sensible, the earlier weirdness was specifically the")
print("bucket-0 sparse-endgame region being undertrained, not a remaining bug in the fix.")
print("=" * 90)
