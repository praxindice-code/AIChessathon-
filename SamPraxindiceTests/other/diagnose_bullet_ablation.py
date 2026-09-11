import chess
import position
import board as board_mod
import nnue as bn


# ============================================================
# NNUE EVALUATION
# ============================================================

def eval_board(board):
    bb, st, mb = position.from_board(board)

    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)

    side = st[0]
    piece_count = board_mod.popcount(bb[0] | bb[1])

    return bn.nnue_eval(
        acc,
        psqt,
        side,
        piece_count,
        0,
    )


# ============================================================
# PRINTING
# ============================================================

def evaluate(label, board, baseline):
    value = eval_board(board)
    delta = value - baseline

    print(
        f"{label:<35}"
        f"eval = {value:>9.4f}   "
        f"delta = {delta:>+9.4f}"
    )

    return value


# ============================================================
# BUILD BOARD
# ============================================================

def make_board(
    turn,
    white_pawns=0,
    white_knights=0,
    white_bishops=0,
    white_rooks=0,
    white_queens=0,
    black_pawns=0,
    black_knights=0,
    black_bishops=0,
    black_rooks=0,
    black_queens=0,
):
    board = chess.Board(None)

    # Kings
    board.set_piece_at(
        chess.E1,
        chess.Piece(chess.KING, chess.WHITE)
    )

    board.set_piece_at(
        chess.E8,
        chess.Piece(chess.KING, chess.BLACK)
    )

    # White pieces
    white_squares = [
        chess.A2, chess.B2, chess.C2, chess.D2,
        chess.E2, chess.F2, chess.G2, chess.H2,
    ]

    for square in white_squares[:white_pawns]:
        board.set_piece_at(
            square,
            chess.Piece(chess.PAWN, chess.WHITE)
        )

    if white_knights:
        board.set_piece_at(
            chess.B1,
            chess.Piece(chess.KNIGHT, chess.WHITE)
        )

    if white_bishops:
        board.set_piece_at(
            chess.C1,
            chess.Piece(chess.BISHOP, chess.WHITE)
        )

    if white_rooks:
        board.set_piece_at(
            chess.A1,
            chess.Piece(chess.ROOK, chess.WHITE)
        )

    if white_queens:
        board.set_piece_at(
            chess.D1,
            chess.Piece(chess.QUEEN, chess.WHITE)
        )

    # Black pieces
    black_squares = [
        chess.A7, chess.B7, chess.C7, chess.D7,
        chess.E7, chess.F7, chess.G7, chess.H7,
    ]

    for square in black_squares[:black_pawns]:
        board.set_piece_at(
            square,
            chess.Piece(chess.PAWN, chess.BLACK)
        )

    if black_knights:
        board.set_piece_at(
            chess.B8,
            chess.Piece(chess.KNIGHT, chess.BLACK)
        )

    if black_bishops:
        board.set_piece_at(
            chess.F8,
            chess.Piece(chess.BISHOP, chess.BLACK)
        )

    if black_rooks:
        board.set_piece_at(
            chess.A8,
            chess.Piece(chess.ROOK, chess.BLACK)
        )

    if black_queens:
        board.set_piece_at(
            chess.D8,
            chess.Piece(chess.QUEEN, chess.BLACK)
        )

    board.turn = turn
    board.castling_rights = 0
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1

    return board


# ============================================================
# TEST 1
# WHITE TO MOVE
# ============================================================

print()
print("=" * 80)
print("CONTROLLED MATERIAL TEST — WHITE TO MOVE")
print("=" * 80)
print()

# Start with kings only.
base_white = make_board(chess.WHITE)

baseline_white = eval_board(base_white)

print(f"Kings only:                       eval = {baseline_white:>9.4f}")
print()

print("ADDING WHITE MATERIAL")
print("-" * 80)

evaluate(
    "White + 1 pawn",
    make_board(chess.WHITE, white_pawns=1),
    baseline_white,
)

evaluate(
    "White + 1 knight",
    make_board(chess.WHITE, white_knights=1),
    baseline_white,
)

evaluate(
    "White + 1 bishop",
    make_board(chess.WHITE, white_bishops=1),
    baseline_white,
)

evaluate(
    "White + 1 rook",
    make_board(chess.WHITE, white_rooks=1),
    baseline_white,
)

evaluate(
    "White + 1 queen",
    make_board(chess.WHITE, white_queens=1),
    baseline_white,
)

print()
print("ADDING BLACK MATERIAL")
print("-" * 80)

evaluate(
    "Black + 1 pawn",
    make_board(chess.WHITE, black_pawns=1),
    baseline_white,
)

evaluate(
    "Black + 1 knight",
    make_board(chess.WHITE, black_knights=1),
    baseline_white,
)

evaluate(
    "Black + 1 bishop",
    make_board(chess.WHITE, black_bishops=1),
    baseline_white,
)

evaluate(
    "Black + 1 rook",
    make_board(chess.WHITE, black_rooks=1),
    baseline_white,
)

evaluate(
    "Black + 1 queen",
    make_board(chess.WHITE, black_queens=1),
    baseline_white,
)


# ============================================================
# TEST 2
# BLACK TO MOVE
# ============================================================

print()
print("=" * 80)
print("CONTROLLED MATERIAL TEST — BLACK TO MOVE")
print("=" * 80)
print()

base_black = make_board(chess.BLACK)

baseline_black = eval_board(base_black)

print(f"Kings only:                       eval = {baseline_black:>9.4f}")
print()

print("ADDING BLACK MATERIAL")
print("-" * 80)

evaluate(
    "Black + 1 pawn",
    make_board(chess.BLACK, black_pawns=1),
    baseline_black,
)

evaluate(
    "Black + 1 knight",
    make_board(chess.BLACK, black_knights=1),
    baseline_black,
)

evaluate(
    "Black + 1 bishop",
    make_board(chess.BLACK, black_bishops=1),
    baseline_black,
)

evaluate(
    "Black + 1 rook",
    make_board(chess.BLACK, black_rooks=1),
    baseline_black,
)

evaluate(
    "Black + 1 queen",
    make_board(chess.BLACK, black_queens=1),
    baseline_black,
)

print()
print("ADDING WHITE MATERIAL")
print("-" * 80)

evaluate(
    "White + 1 pawn",
    make_board(chess.BLACK, white_pawns=1),
    baseline_black,
)

evaluate(
    "White + 1 knight",
    make_board(chess.BLACK, white_knights=1),
    baseline_black,
)

evaluate(
    "White + 1 bishop",
    make_board(chess.BLACK, white_bishops=1),
    baseline_black,
)

evaluate(
    "White + 1 rook",
    make_board(chess.BLACK, white_rooks=1),
    baseline_black,
)

evaluate(
    "White + 1 queen",
    make_board(chess.BLACK, white_queens=1),
    baseline_black,
)


# ============================================================
# END
# ============================================================

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print()
print("Expected general behavior:")
print()
print("WHITE TO MOVE:")
print("  White material should generally increase evaluation.")
print("  Black material should generally decrease evaluation.")
print()
print("BLACK TO MOVE:")
print("  Black material should generally increase evaluation.")
print("  White material should generally decrease evaluation.")
print()
print("The exact numbers do NOT need to resemble centipawns.")
print("We are checking the direction and relative magnitude.")
print()