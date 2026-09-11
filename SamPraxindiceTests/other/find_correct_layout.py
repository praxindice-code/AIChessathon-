import numpy as np
import chess
import position
import board as board_mod
from bullet_forward import screlu, choose_output_bucket

NUM_INPUTS = 768
L0_HIDDEN = 1024
NUM_BUCKETS = 8
L1_INPUT = 2 * L0_HIDDEN

raw = np.fromfile("raw.bin", dtype="<f4")

l0w_size = NUM_INPUTS * L0_HIDDEN
l0b_size = L0_HIDDEN
l1w_size = NUM_BUCKETS * L1_INPUT
l1b_size = NUM_BUCKETS

l0w_flat = raw[0:l0w_size]
l0b = raw[l0w_size : l0w_size + l0b_size]
l1w_flat = raw[l0w_size + l0b_size : l0w_size + l0b_size + l1w_size]
l1b = raw[l0w_size + l0b_size + l1w_size :]


def feature_index_friendly_first(is_white_pov, sq, piece_type, piece_is_white):
    friendly = piece_is_white if is_white_pov else not piece_is_white
    base = (piece_type - 1) if friendly else (piece_type - 1 + 6)
    used_sq = sq if is_white_pov else (sq ^ 56)
    return 64 * base + used_sq


def feature_index_enemy_first(is_white_pov, sq, piece_type, piece_is_white):
    friendly = piece_is_white if is_white_pov else not piece_is_white
    base = (piece_type - 1 + 6) if friendly else (piece_type - 1)  # swapped vs above
    used_sq = sq if is_white_pov else (sq ^ 56)
    return 64 * base + used_sq


def eval_with_variant(py_board, l0w, l1w, feature_fn):
    white_acc = l0b.copy()
    black_acc = l0b.copy()
    for sq in range(64):
        piece = py_board.piece_at(sq)
        if piece is None:
            continue
        w_idx = feature_fn(True, sq, piece.piece_type, piece.color)
        b_idx = feature_fn(False, sq, piece.piece_type, piece.color)
        white_acc += l0w[w_idx]
        black_acc += l0w[b_idx]

    side = 0 if py_board.turn == chess.WHITE else 1
    stm_acc = white_acc if side == 0 else black_acc
    ntm_acc = black_acc if side == 0 else white_acc
    hidden = np.concatenate([screlu(stm_acc), screlu(ntm_acc)])
    bucket = choose_output_bucket(py_board)
    return float(np.dot(l1w[bucket], hidden) + l1b[bucket])


# candidate layouts for l0w: (768,1024) row-major [current] vs (1024,768) row-major then transposed
l0w_variant_A = l0w_flat.reshape(NUM_INPUTS, L0_HIDDEN)  # current assumption
l0w_variant_B = l0w_flat.reshape(L0_HIDDEN, NUM_INPUTS).T  # alternate: swapped declared shape

# candidate layouts for l1w: (8,2048) row-major [current] vs (2048,8) row-major then transposed
l1w_variant_A = l1w_flat.reshape(NUM_BUCKETS, L1_INPUT)  # current assumption
l1w_variant_B = l1w_flat.reshape(L1_INPUT, NUM_BUCKETS).T  # alternate

with_queen = chess.Board("rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2")
without_queen = with_queen.copy()
without_queen.remove_piece_at(chess.D8)
startpos = chess.Board()

variants = [
    ("A: l0w=(768,1024) row-major, l1w=(8,2048) row-major, friendly-first [current impl]", l0w_variant_A, l1w_variant_A, feature_index_friendly_first),
    ("B: l0w swapped+T, l1w=(8,2048) row-major, friendly-first", l0w_variant_B, l1w_variant_A, feature_index_friendly_first),
    ("C: l0w=(768,1024) row-major, l1w swapped+T, friendly-first", l0w_variant_A, l1w_variant_B, feature_index_friendly_first),
    ("D: l0w swapped+T, l1w swapped+T, friendly-first", l0w_variant_B, l1w_variant_B, feature_index_friendly_first),
    ("E: l0w=(768,1024) row-major, l1w=(8,2048) row-major, enemy-first", l0w_variant_A, l1w_variant_A, feature_index_enemy_first),
    ("F: l0w swapped+T, l1w=(8,2048) row-major, enemy-first", l0w_variant_B, l1w_variant_A, feature_index_enemy_first),
    ("G: l0w=(768,1024) row-major, l1w swapped+T, enemy-first", l0w_variant_A, l1w_variant_B, feature_index_enemy_first),
    ("H: l0w swapped+T, l1w swapped+T, enemy-first", l0w_variant_B, l1w_variant_B, feature_index_enemy_first),
]

print(f"{'variant':75} {'startpos':>10} {'delta':>10}")
print(f"{'(want: 0.1236)':75} {'':>10} {'(want -2.5086)':>10}")
for name, l0w, l1w, feat_fn in variants:
    raw_startpos = eval_with_variant(startpos, l0w, l1w, feat_fn)
    raw_with = eval_with_variant(with_queen, l0w, l1w, feat_fn)
    raw_without = eval_with_variant(without_queen, l0w, l1w, feat_fn)
    diff = raw_without - raw_with
    startpos_close = abs(raw_startpos - 0.1236) < 0.5
    delta_close = abs(diff - (-2.5086)) < 0.8
    marker = "  <-- BOTH close!" if (startpos_close and delta_close) else ("  <-- startpos close" if startpos_close else ("  <-- delta close" if delta_close else ""))
    print(f"{name:75} {raw_startpos:10.4f} {diff:10.4f}{marker}")

print()
print("Looking for a variant where BOTH columns are close to Bullet's real values --")
print("a variant matching only one of the two is not yet the right answer.")
