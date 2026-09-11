import chess
import position
import board as board_mod
import nnue as bn
import numpy as np

positions = [
    ("startpos", chess.STARTING_FEN),
    ("white up a full queen, populated middlegame", "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 2 2"),
]

for name, fen in positions:
    py_board = chess.Board(fen)
    bb, st, mb = position.from_board(py_board)
    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)

    white_acc = acc[0, 0]
    black_acc = acc[0, 1]

    print(f"--- {name} ---")
    print(f"white_acc: min={white_acc.min():.3f} max={white_acc.max():.3f} mean={white_acc.mean():.3f}")
    print(f"black_acc: min={black_acc.min():.3f} max={black_acc.max():.3f} mean={black_acc.mean():.3f}")
    frac_above_1_white = np.mean(white_acc > 1.0)
    frac_below_0_white = np.mean(white_acc < 0.0)
    print(f"white_acc: fraction > 1.0 (will saturate to ceiling): {frac_above_1_white:.1%}")
    print(f"white_acc: fraction < 0.0 (will saturate to floor):   {frac_below_0_white:.1%}")
    print()

print("If most values are far outside [0,1] (fraction saturating close to 100%), SCReLU's")
print("clip range is likely wrong for this network's actual scale, or something upstream")
print("(weight loading, feature indexing) is producing inflated accumulator values.")
