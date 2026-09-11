import chess
import position
import board as board_mod
import nnue as bn

positions = [
    ("startpos", chess.STARTING_FEN),
    ("white up a queen", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"),
    ("black up a queen", "3qk3/8/8/8/8/8/8/4K3 w - - 0 1"),
]

for name, fen in positions:
    py_board = chess.Board(fen)
    bb, st, mb = position.from_board(py_board)
    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)
    side = st[0]
    piece_count = board_mod.popcount(bb[0] | bb[1])
    raw = bn.nnue_eval(acc, psqt, side, piece_count, 0)
    print(f"{name:25} raw output: {raw:.4f}   (x600 = {raw*600:.1f})")

print()
print("startpos should be small (near 0). A clean queen-up position should be roughly")
print("+900 to +1200 after whatever the correct multiplier turns out to be -- compare the")
print("'raw' column's ratio between startpos and queen-up to work out the right scale.")
