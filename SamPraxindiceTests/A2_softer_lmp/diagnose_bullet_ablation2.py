import chess
import position
import board as board_mod
import nnue as bn

def eval_board(py_board):
    bb, st, mb = position.from_board(py_board)
    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)
    side = st[0]
    piece_count = board_mod.popcount(bb[0] | bb[1])
    return bn.nnue_eval(acc, psqt, side, piece_count, 0)

# after 1.e4 e5 2.Qh5 -- both queens still on board, black to move
b1 = chess.Board("rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2")
raw_with_queen = eval_board(b1)
print("black queen square:", b1.piece_at(chess.D8))

b2 = b1.copy()
b2.remove_piece_at(chess.D8)  # confirmed real black queen square in THIS position
print("after removal, black queen square:", b2.piece_at(chess.D8))
raw_without_queen = eval_board(b2)

print()
print(f"black to move, black HAS its queen:   raw = {raw_with_queen:.4f}")
print(f"black to move, black's queen REMOVED: raw = {raw_without_queen:.4f}")
print(f"difference: {raw_without_queen - raw_with_queen:.4f}")
print()
print("black is to move in both, and this network's convention is 'positive = good for")
print("whoever is about to move'. Removing black's own queen should make raw_without_queen")
print("clearly LOWER than raw_with_queen if the network learned anything about queen value.")
