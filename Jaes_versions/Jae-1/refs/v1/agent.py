"""The submission entrypoint. The platform imports this file and calls get_move.

Everything expensive happens here at import: numba compiles the whole search, the magic attack
tables are built, and short real searches run so that no compilation is left to happen on the
clock.

The engine underneath is in board.py (position and move generation), evaluate.py (tapered
evaluation) and search.py (alpha-beta). python-chess is used only at the edges: it reads the
FEN we are handed and it checks the move we hand back is legal in that exact position.
"""

import sys
import time

import chess
import numpy as np

import search
from board import zobrist
from evaluate import MATE, MATE_IN_MAX
from position import from_board, to_uci

# The clock is 120 s plus 0.5 s a move on wall time and a flag is a loss, so every number below
# is deliberately conservative: an extra ply is worth far less than a forfeited game.
INCREMENT_MS = 500
OVERHEAD_MS = 90
MIN_MOVES_TO_GO = 16
MAX_MOVES_TO_GO = 30
PANIC_MS = 120
MAX_DEPTH = 63
HISTORY_CAP = 1024

BBS = np.zeros((search.STACK, 8), dtype=np.int64)
STS = np.zeros((search.STACK, 6), dtype=np.int64)
MBS = np.zeros((search.STACK, 64), dtype=np.int8)
MEM = search.new_memory()
TT = search.new_table()


def _same_position(left: chess.Board, right: chess.Board) -> bool:
    """Same placement, side, rights and en passant target; the clocks may differ."""
    return (
        left.board_fen() == right.board_fen()
        and left.turn == right.turn
        and left.castling_rights == right.castling_rights
        and left.ep_square == right.ep_square
    )


def _key(board: chess.Board) -> int:
    bb, st, _ = from_board(board)
    return int(zobrist(bb, st))


class Game:
    """Follows the game across calls so the search knows which positions already occurred.

    The platform only ever hands us a FEN, but the referee claims threefold repetition on its
    own. Replaying the opponent's reply onto the board we kept gives us the real position
    history, which is what turns "this line repeats" into a score the search can act on.
    """

    def __init__(self) -> None:
        self.board: chess.Board | None = None
        self.keys: list[int] = []
        self.moves_played = 0

    def sync(self, board: chess.Board) -> None:
        if self.board is not None:
            if _same_position(self.board, board):
                return
            for move in self.board.legal_moves:
                self.board.push(move)
                if _same_position(self.board, board):
                    self.keys.append(_key(self.board))
                    self._trim()
                    return
                self.board.pop()
        self.board = chess.Board(board.fen())
        self.keys = [_key(self.board)]

    def commit(self, move: chess.Move) -> None:
        self.moves_played += 1
        if self.board is None:
            return
        self.board.push(move)
        self.keys.append(_key(self.board))
        self._trim()

    def _trim(self) -> None:
        if len(self.keys) > HISTORY_CAP:
            self.keys = self.keys[-HISTORY_CAP:]


GAME = Game()


def _budget_ms(time_left_ms: int, moves_played: int) -> tuple[float, float]:
    """Soft and hard limits in milliseconds.

    The soft limit decides whether another iteration is worth starting; the hard limit aborts
    the one already running. Dividing the clock rather than spending a constant means the
    budget shrinks on its own as the clock does, which is what keeps us off the flag.
    """
    remaining = time_left_ms - OVERHEAD_MS
    if remaining <= PANIC_MS:
        return 0.0, 0.0
    moves_to_go = MAX_MOVES_TO_GO - moves_played // 4
    if moves_to_go < MIN_MOVES_TO_GO:
        moves_to_go = MIN_MOVES_TO_GO
    optimum = remaining / moves_to_go + INCREMENT_MS * 0.8
    # an iteration costs roughly twice the one before it, so stopping at a little over half the
    # target is what makes the average spend land on the target rather than well past it
    soft = optimum * 0.55
    hard = optimum * 3.0
    ceiling = remaining * 0.30
    if hard > ceiling:
        hard = ceiling
    if soft > hard * 0.5:
        soft = hard * 0.5
    return soft, hard


def _fallback(board: chess.Board, reason: str) -> str:
    """A legal move with no search behind it. Only reached if something above went wrong."""
    print(f"FALLBACK ({reason}) at {board.fen()}", file=sys.stderr, flush=True)
    best = None
    best_value = -1
    for move in board.legal_moves:
        value = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            value = 100 if victim is None else 100 * victim
        if move.promotion:
            value += 500
        if value > best_value:
            best_value = value
            best = move
    return "0000" if best is None else best.uci()


def _load(board: chess.Board) -> None:
    bb, st, mb = from_board(board)
    BBS[0] = bb
    STS[0] = st
    MBS[0] = mb
    STS[0, 4] = zobrist(bb, st)


def _load_history(keys: list[int]) -> None:
    """Copy the game's position keys into the search path so repetitions are visible."""
    root_key = int(STS[0, 4])
    history = keys if keys else [root_key]
    base = len(history) - 1
    for offset, key in enumerate(history):
        MEM[search.OFF_PATH + offset] = key
    MEM[search.OFF_PATH + base] = root_key
    MEM[search.OFF_CTL + search.CTL_BASE] = base


def _report(uci: str, elapsed_ms: float, soft_ms: float, hard_ms: float) -> None:
    score = int(MEM[search.OFF_CTL + search.CTL_BEST_SCORE])
    depth = int(MEM[search.OFF_CTL + search.CTL_DEPTH])
    nodes = int(MEM[search.OFF_CTL + search.CTL_NODES])
    if score > MATE_IN_MAX or score < -MATE_IN_MAX:
        plies = MATE - abs(score)
        detail = f"mate {((plies + 1) // 2) * (1 if score > 0 else -1)}"
    else:
        detail = f"cp {score}"
    rate = nodes / elapsed_ms if elapsed_ms > 0 else 0.0
    print(
        f"depth {depth} {detail} nodes {nodes} {rate:.0f}knps time {elapsed_ms:.0f}ms "
        f"budget {soft_ms:.0f}/{hard_ms:.0f}ms move {uci}",
        file=sys.stderr,
        flush=True,
    )


def _choose(board: chess.Board, time_left_ms: int) -> str:
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        return legal[0].uci()

    soft_ms, hard_ms = _budget_ms(time_left_ms, GAME.moves_played)
    if hard_ms <= 0.0:
        return _fallback(board, "no time")

    _load(board)
    _load_history(GAME.keys)
    # last move's history is a hint, not a fact; halve it so it fades instead of hardening
    MEM[search.OFF_HISTORY : search.OFF_HISTORY + 8192] //= 2

    started = time.perf_counter()
    MEM[search.OFF_CTL + search.CTL_HARD_NS] = int((started + hard_ms / 1000.0) * 1e9)
    MEM[search.OFF_CTL + search.CTL_SOFT_NS] = int((started + soft_ms / 1000.0) * 1e9)
    packed = int(search.think(BBS, STS, MBS, MEM, TT, MAX_DEPTH))
    if packed == 0:
        return _fallback(board, "empty search")

    uci = to_uci(packed)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return _fallback(board, "unparsable move")
    if move not in board.legal_moves:
        return _fallback(board, f"illegal {uci}")

    _report(uci, (time.perf_counter() - started) * 1000.0, soft_ms, hard_ms)
    return uci


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return "0000"
    try:
        GAME.sync(board)
        uci = _choose(board, time_left_ms)
        GAME.commit(chess.Move.from_uci(uci))
        return uci
    except Exception as failure:
        print(f"search failed, falling back: {failure!r}", file=sys.stderr, flush=True)
        try:
            return _fallback(board, "exception")
        except Exception:
            return "0000"


def _warm_up() -> None:
    """Compile and exercise every jitted path inside the 60 second import budget."""
    positions = [
        chess.STARTING_FEN,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",
        "8/5k2/8/8/8/8/2Q5/4K3 w - - 0 1",
    ]
    for fen in positions:
        board = chess.Board(fen)
        _load(board)
        _load_history([int(STS[0, 4])])
        started = time.perf_counter()
        MEM[search.OFF_CTL + search.CTL_HARD_NS] = int((started + 0.30) * 1e9)
        MEM[search.OFF_CTL + search.CTL_SOFT_NS] = int((started + 0.18) * 1e9)
        search.think(BBS, STS, MBS, MEM, TT, 8)
    # nothing learned while warming up should reach the real game
    TT[:] = 0
    MEM[search.OFF_HISTORY : search.OFF_COUNTER + 8192] = 0


_started = time.perf_counter()
_warm_up()
print(f"engine ready in {time.perf_counter() - _started:.1f}s", file=sys.stderr, flush=True)
