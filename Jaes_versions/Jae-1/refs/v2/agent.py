"""The submission entrypoint. The platform imports this file and calls get_move.

Everything expensive happens here at import: numba compiles the whole search, the magic attack
tables are built, and short real searches run so that no compilation is left to happen on the
clock.

The engine underneath is in board.py (position and move generation), evaluate.py (tapered
evaluation) and search.py (alpha-beta). python-chess is used only at the edges: it reads the
FEN we are handed and it checks the move we hand back is legal in that exact position.
"""

import sys
import threading
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
# four times the documented increment: room for a rule change, not for a runaway budget
MAX_INCREMENT_MS = 2000.0
OVERHEAD_MS = 90
MIN_MOVES_TO_GO = 16
MAX_MOVES_TO_GO = 30
PANIC_MS = 12
MAX_DEPTH = 63
HISTORY_CAP = 1024

BBS = np.zeros((search.STACK, 8), dtype=np.int64)
STS = np.zeros((search.STACK, 6), dtype=np.int64)
MBS = np.zeros((search.STACK, 64), dtype=np.int8)
MEM = search.new_memory()
TT = search.new_table()

# The pondering thread searches with its own stacks and arena but shares the transposition
# table, which is the whole point of it. It is always stopped and joined before the main
# search starts, so the table is never written by two threads at once.
PONDER_SECONDS = 25.0
PONDER_JOIN_S = 0.5
PONDER_ABANDON_S = 5.0
PONDER_BBS = np.zeros((search.STACK, 8), dtype=np.int64)
PONDER_STS = np.zeros((search.STACK, 6), dtype=np.int64)
PONDER_MBS = np.zeros((search.STACK, 64), dtype=np.int8)
PONDER_MEM = search.new_memory()
_PONDER_THREAD: threading.Thread | None = None
_PONDER_ALLOWED = True


def _signature(board: chess.Board) -> tuple[str, bool, int, int | None]:
    """What makes two positions the same position: placement, side, rights, en passant.

    The en passant square only counts when a capture there is actually available. FEN writers
    disagree about whether to record a square nobody can use, and a mismatch there would make
    us throw away the game history we are tracking for no reason.
    """
    ep = board.ep_square if board.has_legal_en_passant() else None
    return board.board_fen(), board.turn, board.castling_rights, ep


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
        # The increment is documented as 0.5 s, but budgeting on a number nobody confirmed is
        # how an engine flags when the number changes. We start from the documented value and
        # replace it with what the clock actually hands back.
        self.increment_ms = float(INCREMENT_MS)
        self.samples: list[float] = []
        self.last_clock: float | None = None
        self.last_spent: float | None = None

    def observe_clock(self, time_left_ms: int) -> None:
        """Infer the real increment from how much the clock grew since our last move."""
        if self.last_clock is not None and self.last_spent is not None:
            gained = time_left_ms - (self.last_clock - self.last_spent)
            # our own stopwatch misses the referee's framing overhead, so `gained` is an
            # underestimate; taking the smallest sample keeps the error on the safe side, and
            # the ceiling stops a clock that behaves unexpectedly from inflating the budget
            if -100.0 <= gained <= MAX_INCREMENT_MS:
                self.samples.append(max(0.0, gained))
                self.increment_ms = min(self.samples[-8:])
        self.last_clock = float(time_left_ms)

    def record_spend(self, elapsed_ms: float) -> None:
        self.last_spent = elapsed_ms

    def sync(self, board: chess.Board) -> None:
        if self.board is not None:
            target = _signature(board)
            if _signature(self.board) == target:
                return
            for move in self.board.legal_moves:
                self.board.push(move)
                if _signature(self.board) == target:
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


def _budget_ms(time_left_ms: int, moves_played: int, increment_ms: float) -> tuple[float, float]:
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
    optimum = remaining / moves_to_go + increment_ms * 0.8
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
    settled = "steady" if MEM[search.OFF_CTL + search.CTL_STABLE] else "changed"
    print(
        f"depth {depth} {detail} nodes {nodes} {rate:.0f}knps time {elapsed_ms:.0f}ms "
        f"budget {soft_ms:.0f}/{hard_ms:.0f}ms inc {GAME.increment_ms:.0f}ms "
        f"root {settled} move {uci}",
        file=sys.stderr,
        flush=True,
    )


def _choose(board: chess.Board, time_left_ms: int) -> str:
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        return legal[0].uci()

    soft_ms, hard_ms = _budget_ms(time_left_ms, GAME.moves_played, GAME.increment_ms)
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


def _stop_ponder() -> None:
    """Bring the pondering thread down before we touch the table it is writing to."""
    global _PONDER_THREAD, _PONDER_ALLOWED
    thread = _PONDER_THREAD
    _PONDER_THREAD = None
    if thread is None:
        return
    PONDER_MEM[search.OFF_CTL + search.CTL_STOP] = 1
    PONDER_MEM[search.OFF_CTL + search.CTL_HARD_NS] = 0
    thread.join(PONDER_JOIN_S)
    if thread.is_alive():
        # it should stop within a node check, so this means something is wrong with the
        # mechanism rather than with this move; wait it out and never ponder again
        thread.join(PONDER_ABANDON_S)
        _PONDER_ALLOWED = False
        print("pondering did not stop promptly, disabling it", file=sys.stderr, flush=True)


def _start_ponder() -> None:
    """Search the position we just moved into, on the opponent's clock.

    The rules allow it: the process keeps its core while the opponent thinks. We do not try to
    guess their reply. Searching the position with them to move fills the shared transposition
    table with the subtree below every one of their candidate moves, so whichever they choose,
    our next search starts from a table that already knows most of what is under it.
    """
    global _PONDER_THREAD
    if not _PONDER_ALLOWED or GAME.board is None:
        return
    board = GAME.board
    if board.is_game_over(claim_draw=False):
        return
    bb, st, mb = from_board(board)
    PONDER_BBS[0] = bb
    PONDER_STS[0] = st
    PONDER_MBS[0] = mb
    PONDER_STS[0, 4] = zobrist(bb, st)

    history = GAME.keys if GAME.keys else [int(PONDER_STS[0, 4])]
    base = len(history) - 1
    for offset, key in enumerate(history):
        PONDER_MEM[search.OFF_PATH + offset] = key
    PONDER_MEM[search.OFF_PATH + base] = int(PONDER_STS[0, 4])
    PONDER_MEM[search.OFF_CTL + search.CTL_BASE] = base

    PONDER_MEM[search.OFF_CTL + search.CTL_STOP] = 0
    deadline = time.perf_counter() + PONDER_SECONDS
    PONDER_MEM[search.OFF_CTL + search.CTL_HARD_NS] = int(deadline * 1e9)
    PONDER_MEM[search.OFF_CTL + search.CTL_SOFT_NS] = int(deadline * 1e9)
    thread = threading.Thread(
        target=search.think,
        args=(PONDER_BBS, PONDER_STS, PONDER_MBS, PONDER_MEM, TT, MAX_DEPTH),
        daemon=True,
    )
    thread.start()
    _PONDER_THREAD = thread


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    global _PONDER_ALLOWED
    entered = time.perf_counter()
    try:
        _stop_ponder()
    except Exception as failure:
        # never let the optional half of the engine take the move with it
        _PONDER_ALLOWED = False
        print(f"stopping the ponder failed, disabling it: {failure!r}", file=sys.stderr)
    try:
        board = chess.Board(fen)
    except ValueError:
        return "0000"
    try:
        GAME.observe_clock(time_left_ms)
        GAME.sync(board)
        uci = _choose(board, time_left_ms)
        GAME.commit(chess.Move.from_uci(uci))
        GAME.record_spend((time.perf_counter() - entered) * 1000.0)
    except Exception as failure:
        print(f"search failed, falling back: {failure!r}", file=sys.stderr, flush=True)
        try:
            return _fallback(board, "exception")
        except Exception:
            return "0000"
    # the move is already decided; pondering is a bonus and must never cost us that move
    try:
        _start_ponder()
    except Exception as failure:
        _PONDER_ALLOWED = False
        print(f"starting the ponder failed, disabling it: {failure!r}", file=sys.stderr)
    return uci


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
    # the pondering thread is the one path a game can hit that the searches above cannot,
    # so start and stop it once here, while there is still an import budget to absorb it
    GAME.sync(chess.Board())
    _start_ponder()
    _stop_ponder()
    GAME.board = None
    GAME.keys = []

    # nothing learned while warming up should reach the real game
    TT[:] = 0
    MEM[search.OFF_HISTORY : search.OFF_COUNTER + 8192] = 0
    PONDER_MEM[search.OFF_HISTORY : search.OFF_COUNTER + 8192] = 0


_started = time.perf_counter()
_warm_up()
print(f"engine ready in {time.perf_counter() - _started:.1f}s", file=sys.stderr, flush=True)
