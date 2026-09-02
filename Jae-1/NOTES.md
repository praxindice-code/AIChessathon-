# Jae-1 engine

A from-scratch bitboard chess engine for AI Chessathon. No third-party engine, no external
weights: magic move generation, a hand-written tapered evaluation and an alpha-beta search,
all jitted with numba at import time.

## Files that ship

`harness/package.py` zips every `*.py` at the root, so these six are the submission:

| file | what it is |
|---|---|
| `agent.py` | entry point, time management, game history, pondering, legality guard |
| `search.py` | iterative deepening, PVS, transposition table, pruning, quiescence |
| `evaluate.py` | tapered evaluation: material, piece-square tables, mobility, pawn structure, king safety |
| `board.py` | bitboard position, magic move generation, make/unmake, Zobrist |
| `position.py` | the only bridge between python-chess and the engine's arrays |
| `magics.py` | magic multipliers, generated offline by `tools/gen_magics.py` |

Everything else (`tools/`, `refs/`) stays out of the zip and out of the way.

## Running things

This machine has no `make` or `uv`, so the Makefile targets are spelled out here against the
local virtualenv. `.venv` is pinned to the platform's versions: python-chess 1.11.2,
numpy 2.5.2, numba 0.67.0.

```bash
.venv/Scripts/python.exe tools/test_perft.py            # move generation vs python-chess
.venv/Scripts/python.exe tools/test_eval.py 4000        # evaluation colour symmetry
.venv/Scripts/python.exe tools/test_engine.py 8 1500    # legality, clocks, mates, history
.venv/Scripts/python.exe tools/bench.py 12              # fixed-depth speed and node counts
.venv/Scripts/python.exe -m harness.arena --opponent baselines/minimax --games 20 --base-ms 10000
.venv/Scripts/python.exe -m harness.package             # build submission.zip
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m mypy
```

`refs/` holds frozen engines kept as sparring partners: `v1` is the first working version,
`v2` is what was submitted. Freeze a copy before you change anything, then measure against it:

```bash
.venv/Scripts/python.exe -m harness.arena --opponent refs/v2 --games 40 --base-ms 10000 --increment-ms 500
```

Match the arena's increment to what the engine expects, or the time management is what you end
up measuring. The engine infers the real increment from the clock after its first move, so it
adapts, but `refs/v1` predates that and always assumes 0.5 s.

## Things to know before changing it

- **numba freezes module-level arrays read-only.** Every mutable buffer is passed into the
  jitted functions instead. That is why `search.py` takes `mem` and `tt` everywhere.
- **A bare integer literal at a call site costs a whole recompile.** numba types `1` as
  `Literal[int](1)` and specialises on it, so `negamax` was being compiled three times and
  import took 39 s. The `np.int64(...)` locals at the top of `negamax` exist to stop that.
- **`inline="always"` made things worse here**, on both compile time and node rate. It was
  tried and removed; do not add it back without measuring.
- **Measure pruning changes as depth at a fixed time, not as node count.** Guarding move count
  pruning with `not pv_node` looked obviously safer and cost 0.76 ply at a fixed three second
  budget (16.62 against 17.38 over eight positions). It was reverted on that measurement.
- **Right shifts on bitboards must go through `srl`.** Bitboards live in int64, so a bare `>>`
  sign-extends bit 63.
- **The increment is measured, not assumed.** `Game.observe_clock` works out how much the
  clock grew since the last move and budgets on that, taking the smallest recent sample so the
  estimate stays low. The documented 0.5 s is only the value used before the first sample.
- **Re-run `tools/test_perft.py` after any move generation change.** It is the only thing
  standing between a subtle bug and an illegal move that loses a game outright.
