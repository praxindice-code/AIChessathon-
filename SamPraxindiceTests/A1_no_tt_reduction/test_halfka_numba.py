import numpy as np
import chess
import random

import position
import board as board_mod
import nnue as bn

npz = np.load("halfka_raw.npz")
ref_weights = {
    "ft_weight": npz["ft_weight"], "ft_bias": npz["ft_bias"],
    "l1w": npz["l1w"], "l1b": npz["l1b"],
    "l2w": npz["l2w"], "l2b": npz["l2b"],
    "outw": npz["outw"], "outb": npz["outb"],
}

import halfka_reference as ref


def ref_compute_accumulator(bb_numba, is_white_pov, py_board):
    """Wraps halfka_reference.compute_accumulator using the weight dict directly."""
    acc = ref_weights["ft_bias"].copy()
    king_sq = py_board.king(chess.WHITE if is_white_pov else chess.BLACK)
    for sq, piece in py_board.piece_map().items():
        idx = ref.feature_index(is_white_pov, king_sq, sq, piece.piece_type, piece.color == chess.WHITE)
        acc = acc + ref_weights["ft_weight"][idx]
    return acc


def ref_forward(white_acc, black_acc, py_board):
    return ref.forward(
        ref_weights["ft_weight"], ref_weights["ft_bias"],
        ref_weights["l1w"], ref_weights["l1b"],
        np.zeros((32, 1024), dtype=np.float32), np.zeros(32, dtype=np.float32),  # already merged, zero factorized
        ref_weights["l2w"], ref_weights["l2b"],
        ref_weights["outw"], ref_weights["outb"],
        py_board,
    )


# === Test 1: refresh_accumulator matches reference, several positions ===
print("=== Test 1: refresh_accumulator vs reference ===")
test_fens = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
    "8/8/4k3/8/8/4K3/4P3/4R3 w - - 0 1",
    "8/8/4k3/8/8/4K3/4P3/4R3 b - - 0 1",
]
worst = 0.0
for fen in test_fens:
    py_board = chess.Board(fen)
    bb, st, mb = position.from_board(py_board)

    numba_white = np.zeros(bn.ACC_WIDTH, dtype=np.float32)
    numba_black = np.zeros(bn.ACC_WIDTH, dtype=np.float32)
    bn.refresh_accumulator(bb, True, numba_white)
    bn.refresh_accumulator(bb, False, numba_black)

    ref_white = ref_compute_accumulator(bb, True, py_board)
    ref_black = ref_compute_accumulator(bb, False, py_board)

    diff_w = np.max(np.abs(numba_white - ref_white))
    diff_b = np.max(np.abs(numba_black - ref_black))
    worst = max(worst, diff_w, diff_b)
    print(f"  {fen[:40]:40} white_diff={diff_w:.2e} black_diff={diff_b:.2e}")
print(f"worst diff across all: {worst:.2e}")
print()

# === Test 2: forward pass matches reference, same positions ===
print("=== Test 2: forward pass vs reference ===")
worst = 0.0
for fen in test_fens:
    py_board = chess.Board(fen)
    bb, st, mb = position.from_board(py_board)

    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)
    side = int(st[0])
    piece_count = board_mod.popcount(bb[0] | bb[1])
    numba_raw = bn.forward(acc[0, 0], acc[0, 1], side, piece_count)
    numba_score = numba_raw * bn.NNUE2SCORE  # forward() returns raw; nnue_eval() scales separately

    ref_white = ref_compute_accumulator(bb, True, py_board)
    ref_black = ref_compute_accumulator(bb, False, py_board)
    ref_score = ref_forward(ref_white, ref_black, py_board)

    diff = abs(numba_score - ref_score)
    worst = max(worst, diff)
    print(f"  {fen[:40]:40} numba={numba_score:.4f} ref={ref_score:.4f} diff={diff:.2e}")
print(f"worst diff across all: {worst:.2e}")
print()

# === Test 3: incremental update matches full refresh, including king moves ===
print("=== Test 3: incremental nnue_make_move vs full refresh, random games with king moves ===")
STACK = 150
random.seed(7)
worst = 0.0
king_moves_tested = 0

for game_num in range(10):
    py_board = chess.Board()
    bb0, st0, mb0 = position.from_board(py_board)
    bbs = np.zeros((STACK, 8), dtype=np.int64)
    sts = np.zeros((STACK, 6), dtype=np.int64)
    mbs = np.zeros((STACK, 64), dtype=np.int8)
    bbs[0], sts[0], mbs[0] = bb0, st0, mb0

    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bbs[0], acc, psqt)

    moves_buf = np.zeros(STACK * 256, dtype=np.int64)
    for ply in range(60):
        if py_board.is_game_over():
            break
        start = ply * 256
        end = board_mod.gen_moves(bbs[ply], sts[ply], mbs[ply], moves_buf, start)
        legal_packed = []
        for i in range(start, end):
            mv = moves_buf[i]
            uci = position.to_uci(mv)
            try:
                if chess.Move.from_uci(uci) in py_board.legal_moves:
                    legal_packed.append(mv)
            except Exception:
                pass
        if not legal_packed:
            break
        chosen = random.choice(legal_packed)
        move = chess.Move.from_uci(position.to_uci(chosen))

        moved_piece = py_board.piece_at(move.from_square)
        is_king_move = moved_piece is not None and moved_piece.piece_type == chess.KING
        if is_king_move:
            king_moves_tested += 1

        board_mod.make_move(bbs, sts, mbs, ply, chosen)
        bn.nnue_make_move(bbs, mbs, acc, psqt, ply)
        py_board.push(move)

        # cross-check: incremental result vs a full refresh from this exact position
        check_white = np.zeros(bn.ACC_WIDTH, dtype=np.float32)
        check_black = np.zeros(bn.ACC_WIDTH, dtype=np.float32)
        bn.refresh_accumulator(bbs[ply + 1], True, check_white)
        bn.refresh_accumulator(bbs[ply + 1], False, check_black)

        diff_w = np.max(np.abs(acc[ply + 1, 0] - check_white))
        diff_b = np.max(np.abs(acc[ply + 1, 1] - check_black))
        worst = max(worst, diff_w, diff_b)
        if diff_w > 1e-3 or diff_b > 1e-3:
            print(f"  MISMATCH game {game_num} ply {ply} (king_move={is_king_move}): diff_w={diff_w:.4f} diff_b={diff_b:.4f}")

print(f"tested {king_moves_tested} king moves across all games")
print(f"worst incremental-vs-refresh diff: {worst:.2e}")
