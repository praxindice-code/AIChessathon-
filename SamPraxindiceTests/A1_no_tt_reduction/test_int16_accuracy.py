import numpy as np
import chess

import position
import board as board_mod
import halfka_reference as ref

# load int16-quantized weights and dequantize back to float32
qnpz = np.load("halfka_quantized_raw_16bit.npz")
fnpz = np.load("halfka_raw.npz")  # full-precision, for comparison


def dequantize(key):
    return qnpz[key].astype(np.float32) / qnpz[f"{key}_scale"]


dq_weights = {
    "ft_weight": dequantize("ft_weight"), "ft_bias": dequantize("ft_bias"),
    "l1w": dequantize("l1w"), "l1b": dequantize("l1b"),
    "l2w": dequantize("l2w"), "l2b": dequantize("l2b"),
    "outw": dequantize("outw"), "outb": dequantize("outb"),
}
fp_weights = {
    "ft_weight": fnpz["ft_weight"], "ft_bias": fnpz["ft_bias"],
    "l1w": fnpz["l1w"], "l1b": fnpz["l1b"],
    "l2w": fnpz["l2w"], "l2b": fnpz["l2b"],
    "outw": fnpz["outw"], "outb": fnpz["outb"],
}

# quantization error on the weights themselves, before even running a forward pass
for key in fp_weights:
    err = np.abs(fp_weights[key] - dq_weights[key])
    print(f"{key:12} max_abs_error={err.max():.6f}  mean_abs_error={err.mean():.6f}  "
          f"relative to std={fp_weights[key].std():.4f}: {err.mean()/fp_weights[key].std()*100:.2f}%")

print()
print("=== Forward pass: int16-dequantized vs full-precision, real positions ===")

test_fens = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
    "8/8/4k3/8/8/4K3/4P3/4R3 w - - 0 1",
    "8/8/4k3/8/8/4K3/4P3/4R3 b - - 0 1",
]

worst_diff = 0.0
worst_pct = 0.0
for fen in test_fens:
    py_board = chess.Board(fen)

    fp_score = ref.forward(
        fp_weights["ft_weight"], fp_weights["ft_bias"],
        fp_weights["l1w"], fp_weights["l1b"],
        np.zeros((32, 1024), dtype=np.float32), np.zeros(32, dtype=np.float32),
        fp_weights["l2w"], fp_weights["l2b"],
        fp_weights["outw"], fp_weights["outb"], py_board,
    )
    dq_score = ref.forward(
        dq_weights["ft_weight"], dq_weights["ft_bias"],
        dq_weights["l1w"], dq_weights["l1b"],
        np.zeros((32, 1024), dtype=np.float32), np.zeros(32, dtype=np.float32),
        dq_weights["l2w"], dq_weights["l2b"],
        dq_weights["outw"], dq_weights["outb"], py_board,
    )
    diff = abs(fp_score - dq_score)
    pct = diff / max(abs(fp_score), 1.0) * 100
    worst_diff = max(worst_diff, diff)
    worst_pct = max(worst_pct, pct)
    print(f"  {fen[:40]:40} full_precision={fp_score:10.4f}  int16_dequant={dq_score:10.4f}  diff={diff:.4f} ({pct:.2f}%)")

print()
print(f"worst absolute diff (in centipawn-equivalent units): {worst_diff:.4f}")
print(f"worst relative diff: {worst_pct:.2f}%")
