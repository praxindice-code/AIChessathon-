"""Pure-Python reference implementation of this project's HalfKAv2_hm^ network, built directly
from the real nnue-pytorch source (composed_feature_transformer.py, double_ft_functions.py,
layer_stacks.py, halfka_v2_hm.py, quantize.py) -- not guessed or assumed.

Architecture, traced precisely from source:
    1. Per-perspective (white, black) HalfKAv2_hm feature transformer: 1024-wide hidden +
       8-wide PSQT, computed from merged_weight() = weight + virtual_weight.repeat(32, 1)
       (32 king buckets).
    2. Combine perspectives: l0_ = clamp(us*cat[own,enemy] + them*cat[enemy,own], 0, ~0.996)
       -- a genuinely unusual "paired multiplication" activation, not SCReLU:
       split into 4x512 chunks, multiply chunk0*chunk1 and chunk2*chunk3, concat back to 1024.
    3. layer_stacks: l1 (factorized, bucket-selected) -> squared+concat -> clip(~0.992) ->
       l2 (bucket-selected) -> squared+concat -> clip(~0.992) -> output (bucket-selected,
       128-wide input) -> + skip connection (l1's last two neurons, subtracted).
    4. PSQT correction: + (wpsqt - bpsqt) * (us - 0.5)
    5. Final score: output * nnue2score (600.0, the default -- needs confirming this run
       actually used the default rather than a custom value).

Correction factors (l0_correction_factor, sqr_crelu_correction_factor) are both exactly 1.0
under default quantize config, confirmed by direct computation -- safe to omit from the math.
The clip bounds (max_ft_activation, max_hidden_activation) are NOT quantization-only; they're
called unconditionally in the real forward(), so they're kept here as genuine structural steps.
"""

import chess
import numpy as np

NUM_SQ = 64
NUM_PT = 12
NUM_PLANES = NUM_SQ * NUM_PT  # 768
NUM_BUCKETS = NUM_SQ // 2  # 32
NUM_INPUTS = NUM_PLANES * NUM_BUCKETS  # 24576
L1_SIZE = 1024  # per-perspective hidden width (before pairing)
NUM_PSQT_BUCKETS = 8
NUM_LS_BUCKETS = 8

MAX_FT_ACTIVATION = 255.0 / 256.0
MAX_HIDDEN_ACTIVATION = 127.0 / 128.0
NNUE2SCORE = 600.0  # default -- confirm this matches the actual training run

# fmt: off
KING_BUCKETS = [
  -1, -1, -1, -1, 31, 30, 29, 28,
  -1, -1, -1, -1, 27, 26, 25, 24,
  -1, -1, -1, -1, 23, 22, 21, 20,
  -1, -1, -1, -1, 19, 18, 17, 16,
  -1, -1, -1, -1, 15, 14, 13, 12,
  -1, -1, -1, -1, 11, 10, 9, 8,
  -1, -1, -1, -1, 7, 6, 5, 4,
  -1, -1, -1, -1, 3, 2, 1, 0
]
# fmt: on


def _orient(is_white_pov: bool, sq: int, ksq: int) -> int:
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def feature_index(is_white_pov: bool, king_sq: int, sq: int, piece_type: int, piece_is_white: bool) -> int:
    """Directly ported from halfka_v2_hm.py's _halfka_idx. Note: friendly/enemy is
    INTERLEAVED per piece type (pawn-friendly=0, pawn-enemy=1, knight-friendly=2, ...), NOT
    block-separated like Bullet's Chess768 was -- a real, confirmed structural difference."""
    color_is_white = piece_is_white
    p_idx = (piece_type - 1) * 2 + (color_is_white != is_white_pov)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KING_BUCKETS[o_ksq] * 768


def compute_accumulator(weight: np.ndarray, bias: np.ndarray, board: chess.Board, is_white_pov: bool):
    """weight: (NUM_INPUTS, 1032) merged (real+virtual) weight matrix.
    bias: (1032,) bias vector (only first 1024 are meaningful; last 8 PSQT bias forced to 0).
    Returns raw (pre-clip) 1032-wide accumulator for this perspective."""
    acc = bias.copy()
    king_sq = board.king(chess.WHITE if is_white_pov else chess.BLACK)
    for sq, piece in board.piece_map().items():
        idx = feature_index(is_white_pov, king_sq, sq, piece.piece_type, piece.color == chess.WHITE)
        acc = acc + weight[idx]
    return acc


def forward(weight: np.ndarray, bias: np.ndarray, l1w: np.ndarray, l1b: np.ndarray,
            l1fw: np.ndarray, l1fb: np.ndarray, l2w: np.ndarray, l2b: np.ndarray,
            outw: np.ndarray, outb: np.ndarray, board: chess.Board) -> float:
    side_to_move_white = board.turn == chess.WHITE

    white_acc = compute_accumulator(weight, bias, board, True)
    black_acc = compute_accumulator(weight, bias, board, False)

    w, wpsqt = white_acc[:L1_SIZE], white_acc[L1_SIZE:]
    b, bpsqt = black_acc[:L1_SIZE], black_acc[L1_SIZE:]

    piece_count = len(board.piece_map())
    bucket_idx = (piece_count - 1) // 4
    bucket_idx = max(0, min(NUM_PSQT_BUCKETS - 1, bucket_idx))
    wpsqt_val = wpsqt[bucket_idx]
    bpsqt_val = bpsqt[bucket_idx]

    if side_to_move_white:
        us, them = 1.0, 0.0
    else:
        us, them = 0.0, 1.0

    # Matches double_ft_functions.py exactly: l0_ = us*cat([w,b]) + them*cat([b,w])
    l0_pre = us * np.concatenate([w, b]) + them * np.concatenate([b, w])
    l0_pre = np.clip(l0_pre, 0.0, MAX_FT_ACTIVATION)

    l0_s = np.split(l0_pre, 4)
    l0_ = np.concatenate([l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]])  # 1024-wide

    ls_bucket = (piece_count - 1) // 4
    ls_bucket = max(0, min(NUM_LS_BUCKETS - 1, ls_bucket))

    # l1: FactorizedStackedLinear -- merged weight is linear + factorized, sliced per bucket
    merged_l1w = l1w[ls_bucket * 32:(ls_bucket + 1) * 32] + l1fw
    merged_l1b = l1b[ls_bucket * 32:(ls_bucket + 1) * 32] + l1fb
    l1c = merged_l1w @ l0_ + merged_l1b  # (32,)

    l1x_out = l1c[-2] - l1c[-1]
    l1_sqr = l1c ** 2
    l1x = np.concatenate([l1_sqr, l1c])
    l1x = np.clip(l1x, 0.0, MAX_HIDDEN_ACTIVATION)

    l2_slice_w = l2w[ls_bucket * 32:(ls_bucket + 1) * 32]
    l2_slice_b = l2b[ls_bucket * 32:(ls_bucket + 1) * 32]
    l2c = l2_slice_w @ l1x + l2_slice_b  # (32,)

    l2_sqr = l2c ** 2
    l2x = np.concatenate([l2_sqr, l2c])
    l2x = np.clip(l2x, 0.0, MAX_HIDDEN_ACTIVATION)

    l3_input = np.concatenate([l1x, l2x])  # 128-wide
    out_slice_w = outw[ls_bucket:ls_bucket + 1]
    out_slice_b = outb[ls_bucket:ls_bucket + 1]
    l3c = out_slice_w @ l3_input + out_slice_b  # (1,)

    l3x = l3c[0] + l1x_out
    final = l3x + (wpsqt_val - bpsqt_val) * (us - 0.5)

    return float(final) * NNUE2SCORE
