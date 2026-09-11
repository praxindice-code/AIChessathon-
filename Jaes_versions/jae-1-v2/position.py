"""Bridge between python-chess and the engine's own arrays.

python-chess owns the FEN and the final legality check; everything in between runs on the
numpy arrays described in board.py. Keeping the conversion in one place means there is exactly
one spot where a disagreement between the two representations could start.
"""

import chess
import numpy as np

from board import PAWN_ATTACKS, i64

PROMO_TO_UCI = {0: "", chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q"}


def from_board(board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a python-chess board as (bb, st, mb)."""
    bb = np.zeros(8, dtype=np.int64)
    mb = np.zeros(64, dtype=np.int8)
    for square, piece in board.piece_map().items():
        colour = 0 if piece.color == chess.WHITE else 1
        bb[colour] |= i64(1 << square)
        bb[piece.piece_type + 1] |= i64(1 << square)
        mb[square] = piece.piece_type

    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= 8

    side = 0 if board.turn == chess.WHITE else 1
    ep = -1
    # Only record the square when a pawn can really take there, matching make_move, so the same
    # position always hashes the same way whether we reached it or were handed it.
    if board.ep_square is not None and (
        PAWN_ATTACKS[1 - side][board.ep_square] & bb[2] & bb[side]
    ):
        ep = board.ep_square

    st = np.zeros(6, dtype=np.int64)
    st[0] = side
    st[1] = rights
    st[2] = ep
    st[3] = board.halfmove_clock
    return bb, st, mb


def to_uci(move: int) -> str:
    """Decode a packed move into the UCI string the platform expects."""
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    return chess.square_name(frm) + chess.square_name(to) + PROMO_TO_UCI.get(promo, "")
