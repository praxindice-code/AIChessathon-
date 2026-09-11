"""Quantizes halfka_raw.npz's weight arrays to int16 for file-size reduction ONLY -- all actual
inference computation still happens in float32 (dequantized immediately at nnue.py's import
time), matching the same proven-safe pattern as this project's original weights.qz (int16 +
gzip, ~6.3cp error vs full float32).

This is NOT the same as int8/true-quantized inference (which changes the actual arithmetic and
carries real accumulation-overflow risk, as found earlier in this project). This only shrinks
the file on disk; the math stays identical to the already-validated float32 pipeline.

Run from the same folder as halfka_raw.npz.
"""

import argparse
import gzip
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=[8, 16], default=16)
args = parser.parse_args()

if args.bits == 16:
    SAFE_MAX = 30000.0
    DTYPE = np.int16
    DTYPE_MIN, DTYPE_MAX = -32768, 32767
else:
    SAFE_MAX = 120.0
    DTYPE = np.int8
    DTYPE_MIN, DTYPE_MAX = -128, 127

npz = np.load("halfka_raw.npz")

quantized = {}
scales = {}
for key in npz.files:
    arr = npz[key]
    if key == "nnue2score" or arr.dtype != np.float32:
        quantized[key] = arr
        continue

    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        scale = 1.0
    else:
        scale = SAFE_MAX / max_abs

    q = np.round(arr * scale).astype(DTYPE)
    assert q.min() >= DTYPE_MIN and q.max() <= DTYPE_MAX, f"{key} overflowed {DTYPE} range"

    quantized[key] = q
    scales[key] = np.float32(scale)

np.savez(f"halfka_quantized_raw_{args.bits}bit.npz", **quantized, **{f"{k}_scale": v for k, v in scales.items()})

with open(f"halfka_quantized_raw_{args.bits}bit.npz", "rb") as f:
    raw = f.read()
with gzip.open(f"halfka_quantized_{args.bits}bit.npz.gz", "wb", compresslevel=9) as f:
    f.write(raw)

import os
orig_size = os.path.getsize("halfka_raw.npz")
quant_size = os.path.getsize(f"halfka_quantized_raw_{args.bits}bit.npz")
gz_size = os.path.getsize(f"halfka_quantized_{args.bits}bit.npz.gz")

print(f"[{args.bits}-bit] original (float32):        {orig_size/1e6:.1f} MB")
print(f"[{args.bits}-bit] quantized:                  {quant_size/1e6:.1f} MB  ({orig_size/quant_size:.2f}x smaller)")
print(f"[{args.bits}-bit] quantized + gzip:            {gz_size/1e6:.1f} MB  ({orig_size/gz_size:.2f}x smaller)")
print()
print("scales used per array:")
for k, v in scales.items():
    print(f"  {k}: {v:.2f}")
