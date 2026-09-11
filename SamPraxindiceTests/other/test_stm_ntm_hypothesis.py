import numpy as np
import chess
from bullet_network_loader import load_raw_bin
from bullet_forward import screlu, choose_output_bucket

weights = load_raw_bin("raw.bin")


def feature_index(friendly_is_white, sq, piece_type, piece_is_white, do_flip):
    friendly = piece_is_white if friendly_is_white else not piece_is_white
    base = (piece_type - 1) if friendly else (piece_type - 1 + 6)
    used_sq = (sq ^ 56) if do_flip else sq
    return 64 * base + used_sq


def eval_position(board):
    """Directly computes stm_acc (friendly-first, NEVER flips) and ntm_acc (friendly-first
    from the opponent's own framing, ALWAYS flips) -- matching Bullet's actual rule that the
    flip depends on which slot (stm/ntm), not on a fixed color."""
    stm_is_white = board.turn == chess.WHITE

    stm_acc = weights["l0b"].copy()
    ntm_acc = weights["l0b"].copy()

    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        # stm accumulator: friendly relative to the actual side to move, NEVER flip
        idx_stm = feature_index(stm_is_white, sq, piece.piece_type, piece.color, do_flip=False)
        stm_acc += weights["l0w"][idx_stm]
        # ntm accumulator: friendly relative to the OTHER side, ALWAYS flip
        idx_ntm = feature_index(not stm_is_white, sq, piece.piece_type, piece.color, do_flip=True)
        ntm_acc += weights["l0w"][idx_ntm]

    hidden = np.concatenate([screlu(stm_acc), screlu(ntm_acc)])
    bucket = choose_output_bucket(board)
    return float(np.dot(weights["l1w"][bucket], hidden) + weights["l1b"][bucket])


with_queen = chess.Board("rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2")
without_queen = with_queen.copy()
without_queen.remove_piece_at(chess.D8)

raw_with = eval_position(with_queen)
raw_without = eval_position(without_queen)

print(f"queen present:  {raw_with:.7f}   (Bullet's real value: 0.4403547)")
print(f"queen removed:  {raw_without:.7f}   (Bullet's real value: -2.068251)")
print(f"difference:     {raw_without - raw_with:.7f}   (Bullet's real difference: -2.5086057)")
