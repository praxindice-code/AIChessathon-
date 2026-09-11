"""Chess768 feature indexing + forward pass for the Bullet-trained network
(768 -> 1024x2 -> 8 buckets, SCReLU).

Every piece of math here is sourced, not assumed -- see comments at each step. One thing
flagged honestly rather than glossed over: this implementation flips the square vertically for
whichever side is NOT to move, which is standard practice for shared-weight dual-perspective
networks (necessary for a single weight matrix H to make sense across both perspectives at
all -- without it, "my pawn on e2" and "my pawn on e7" would need to mean the same thing to a
shared H, which they can't). This specific detail wasn't independently confirmed against
Bullet's exact source the way the SCReLU formula and bucket formula were -- it should be
verified once real network weights are available to test end-to-end, e.g. by checking that
a mirrored position (same position, reflected top-to-bottom with colours swapped) evaluates to
the same score, which should hold if this convention is right and will clearly fail if it isn't.
"""

import numpy as np

# from Bullet's own docs (1-basics.md): white_pawn=0, white_knight=1, ..., black_king=11
_PIECE_CODE = {
    (1, True): 0, (2, True): 1, (3, True): 2, (4, True): 3, (5, True): 4, (6, True): 5,      # white
    (1, False): 6, (2, False): 7, (3, False): 8, (4, False): 9, (5, False): 10, (6, False): 11,  # black
}


def feature_indices(board) -> tuple[np.ndarray, np.ndarray]:
    """Returns (stm_indices, ntm_indices): active feature indices (into the 768-wide input)
    from the side-to-move's perspective and the not-to-move side's perspective, for a
    python-chess Board. Each is relabeled friendly=0..5, enemy=6..11 relative to that
    perspective's own side, with the square vertically flipped for whichever perspective is
    NOT the actual side to move on the real board (see module docstring on this convention).
    """
    stm = board.turn  # True = white to move
    stm_idx = []
    ntm_idx = []

    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        piece_type = piece.piece_type
        is_white = piece.color

        # from stm's own perspective: friendly if this piece's colour matches stm
        friendly_for_stm = is_white if stm else (not is_white)
        base = (piece_type - 1) if friendly_for_stm else (piece_type - 1 + 6)
        stm_square = square if stm else (square ^ 56)  # flip vertically if black is stm
        stm_idx.append(64 * base + stm_square)

        # from the OTHER perspective (as if that side were "to move" for indexing purposes)
        friendly_for_ntm = is_white if not stm else (not is_white)
        base2 = (piece_type - 1) if friendly_for_ntm else (piece_type - 1 + 6)
        ntm_square = square if not stm else (square ^ 56)
        ntm_idx.append(64 * base2 + ntm_square)

    return np.array(stm_idx, dtype=np.int64), np.array(ntm_idx, dtype=np.int64)


def screlu(x: np.ndarray) -> np.ndarray:
    """Squared Clipped ReLU: clip(x, 0, 1) ** 2 -- confirmed via two independent sources
    (Chess Programming Wiki's NNUE quantization writeup, and a developer devlog describing
    their own from-scratch implementation: `a = clamp(x, 0, 1); f(x) = a * a * weight`)."""
    return np.clip(x, 0.0, 1.0) ** 2


def choose_output_bucket(board, num_buckets: int = 8) -> int:
    """(total_piece_count - 2) / divisor, where divisor = ceil(32 / num_buckets).
    Confirmed directly from the Chess Programming Wiki's NNUE article, output-bucket section
    (the -2 accounts for the two kings, matching Bullet's material-count-based bucketing)."""
    divisor = (32 + num_buckets - 1) // num_buckets
    piece_count = bin(board.occupied).count("1")
    bucket = (piece_count - 2) // divisor
    return max(0, min(num_buckets - 1, bucket))


def forward(weights: dict, board) -> float:
    """Full (non-incremental) forward pass -- useful for validation before building the
    incremental accumulator version for actual search use."""
    stm_idx, ntm_idx = feature_indices(board)

    acc_stm = weights["l0b"].copy()
    for i in stm_idx:
        acc_stm += weights["l0w"][i]

    acc_ntm = weights["l0b"].copy()
    for i in ntm_idx:
        acc_ntm += weights["l0w"][i]

    hidden = np.concatenate([screlu(acc_stm), screlu(acc_ntm)])  # (2048,)

    bucket = choose_output_bucket(board)
    output = np.dot(weights["l1w"][bucket], hidden) + weights["l1b"][bucket]
    return float(output)
