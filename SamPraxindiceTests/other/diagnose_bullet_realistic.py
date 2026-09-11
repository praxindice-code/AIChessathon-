import chess
import position
import board as board_mod
import nnue as bn

positions = [
    ("startpos", chess.STARTING_FEN),
    ("after e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("kiwipete (roughly balanced, complex)", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("white up a rook, else equal", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKB1R w KQkq - 0 1".replace("r1bqkbnr", "1bqkbnr")),
    ("white up a full queen, populated middlegame", "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 2 2"),
]

for name, fen in positions:
    try:
        py_board = chess.Board(fen)
    except Exception as e:
        print(f"{name}: FEN error {e}")
        continue
    bb, st, mb = position.from_board(py_board)
    acc, psqt = bn.new_accumulator_stack()
    bn.nnue_init_root(bb, acc, psqt)
    side = st[0]
    piece_count = board_mod.popcount(bb[0] | bb[1])
    raw = bn.nnue_eval(acc, psqt, side, piece_count, 0)
    print(f"{name:45} pieces={piece_count:2} stm={'white' if side==0 else 'black'}  raw={raw:.4f}")
