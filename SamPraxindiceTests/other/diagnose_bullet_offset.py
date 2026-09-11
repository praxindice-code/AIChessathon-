import numpy as np
import chess
import position
import board as board_mod
import nnue as bn

positions = [
    ("startpos", chess.STARTING_FEN),
    ("queen test position", "rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"),
]

for name, fen in positions:
    py_board = chess.Board(fen)
    bb, st, mb = position.from_board(py_board)
    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)

    side = int(st[0])
    piece_count = board_mod.popcount(bb[0] | bb[1])
    bucket = (piece_count - 2) // bn.BUCKET_DIVISOR
    bucket = max(0, min(bn.NUM_BUCKETS - 1, bucket))

    stm_acc = acc[0, 0] if side == 0 else acc[0, 1]
    ntm_acc = acc[0, 1] if side == 0 else acc[0, 0]

    hidden = np.concatenate([bn._screlu(stm_acc), bn._screlu(ntm_acc)])
    dot_product = float(np.dot(bn.L1_WEIGHT[bucket], hidden))
    bias_term = float(bn.L1_BIAS[bucket])

    print(f"--- {name} ---")
    print(f"bucket: {bucket}")
    print(f"dot(L1_WEIGHT[bucket], hidden): {dot_product:.4f}")
    print(f"L1_BIAS[bucket]:                {bias_term:.4f}")
    print(f"total (dot + bias):             {dot_product + bias_term:.4f}")
    print()

print("If L1_BIAS[bucket] alone is already large (close to the ~9-12 range we're seeing),")
print("the bias term itself is the dominant contributor to the offset -- worth checking if")
print("L1_BIAS needs different handling (e.g. wrong bucket, or a missing scale factor)")
print("compared to how Bullet actually applies it during its own inference.")
print()
print("Also printing L1_BIAS for ALL 8 buckets, to see if this one is unusually large:")
for b in range(bn.NUM_BUCKETS):
    print(f"  bucket {b}: L1_BIAS = {bn.L1_BIAS[b]:.4f}")
