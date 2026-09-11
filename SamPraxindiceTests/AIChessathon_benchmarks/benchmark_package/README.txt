AIChessathon benchmark package
=============================
Source: uploaded play_match_elo_5600g_3060.py, search.py and agent.py recovered
from Analyze Chess Bot Losses. The subsequently supplied trueAgentV5H search.py
and agent.py match those originals byte-for-byte. reference/ also includes the
local board.py, position.py, magics.py, evaluate.py and matching nnue.py.
Weights/books/tablebases are not bundled. Supply your existing engine folders.

Layout (run commands from this package folder):
  benchmark_common.py                  shared runner and process bridge
  play_match_elo_5600g_3060_fixed.py     agent vs Stockfish
  play_match_head_to_head.py            agent A vs agent B
  openings.txt                         small paired-opening screening suite
  test_benchmarks.py, requirements.txt
  build_variants.py                    reproducible variant generator
  variants/A0/search.py                unchanged baseline
  variants/A1/search.py ... A5/search.py (cumulative; each has changes.diff)
  A6_A7_INSTRUCTIONS.txt               explicit implementation placeholders
  reference/                          original benchmark and engine Python files
  engines/Bullet/                      YOUR complete Bullet engine
  engines/HalfKA/                      YOUR complete HalfKA engine
  engines/A0/ ... engines/A5/           copies of chosen complete engine
  results/                            created by commands below

Each engine folder needs agent.py, search.py, board.py, position.py,
evaluate.py, matching nnue.py, weights and all original dependencies/data.
Use the appropriate NNUE loader and architecture for each network; do not
assume Bullet and HalfKA weight files are interchangeable. Keep all settings
other than the intended experiment identical (including books/tablebases).
Copy variants/A1/search.py to engines/A1/search.py, etc., replacing search.py
only in these experimental engine COPIES. A0 gets variants/A0/search.py.
The supplied agent API is get_move(fen, whole_remaining_clock_ms).

Setup and validation (Windows CMD, Python 3.12):
  python -m pip install -r requirements.txt
  python -m compileall -q .
  python -m unittest -v test_benchmarks
Install your original engine requirements in this same Python environment.

Set your executable once:
  set "SF=C:\Users\punam\coding\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

Smoke tests first:
  python play_match_elo_5600g_3060_fixed.py --engine-a engines/Bullet --stockfish "%SF%" --games 2 --workers 1 --base 10 --increment 0.5 --pgn-out results/smoke_sf.pgn
  python play_match_head_to_head.py --engine-a engines/Bullet --engine-b engines/HalfKA --games 2 --workers 1 --base 10 --increment 0.5 --pgn-out results/smoke_h2h.pgn

NNUE bake-off (same search, clocks, opening file and opponent settings):
  python play_match_elo_5600g_3060_fixed.py --engine-a engines/Bullet --stockfish "%SF%" --games 200 --workers 1 --skill 15 --sf-threads 1 --sf-hash 64 --base 60 --increment 0.5 --pgn-out results/bullet_sf.pgn
  python play_match_elo_5600g_3060_fixed.py --engine-a engines/HalfKA --stockfish "%SF%" --games 200 --workers 1 --skill 15 --sf-threads 1 --sf-hash 64 --base 60 --increment 0.5 --pgn-out results/halfka_sf.pgn
  python play_match_head_to_head.py --engine-a engines/Bullet --engine-b engines/HalfKA --games 200 --workers 1 --base 60 --increment 0.5 --pgn-out results/bullet_halfka.pgn

Cumulative search experiment: compare each step with its immediate predecessor.
  python play_match_head_to_head.py --engine-a engines/A1 --engine-b engines/A0 --games 200 --base 60 --increment 0.5 --pgn-out results/A1_A0.pgn
  python play_match_head_to_head.py --engine-a engines/A2 --engine-b engines/A1 --games 200 --base 60 --increment 0.5 --pgn-out results/A2_A1.pgn
  python play_match_head_to_head.py --engine-a engines/A3 --engine-b engines/A2 --games 200 --base 60 --increment 0.5 --pgn-out results/A3_A2.pgn
  python play_match_head_to_head.py --engine-a engines/A4 --engine-b engines/A3 --games 200 --base 60 --increment 0.5 --pgn-out results/A4_A3.pgn
  python play_match_head_to_head.py --engine-a engines/A5 --engine-b engines/A4 --games 200 --base 60 --increment 0.5 --pgn-out results/A5_A4.pgn
External comparison for every variant (CMD interactive; use %%V in .bat):
  for %V in (A0 A1 A2 A3 A4 A5) do python play_match_elo_5600g_3060_fixed.py --engine-a engines/%V --stockfish "%SF%" --games 200 --skill 15 --base 60 --increment 0.5 --pgn-out results/%V_sf.pgn

Final comparison example if A5 wins (substitute the actual selected variant):
  python play_match_head_to_head.py --engine-a engines/A5 --engine-b engines/A0 --games 500 --workers 1 --base 120 --increment 0.5 --cpu-a 0 --cpu-b 2 --pgn-out results/final_A5_A0.pgn

One-core testing:
--workers 1 controls concurrent GAMES; it does not constrain CPU use by itself.
--cpu-a and --cpu-b pin the two engine processes to one logical CPU each.
Choose allowed CPU IDs on DIFFERENT physical cores; 0 and 2 above are examples,
not a verified mapping of your machine. Do not pin both engines to one CPU.
The bridge sets common numerical-library thread counts to 1 before import.
Stockfish gets --sf-threads 1. Pinning constrains agent ponder threads too.
Pondering is OFF by default using the supplied agent's private hooks; --ponder
enables it for both Python engines. Confirm competition rules before enabling.
CPU affinity does not disable CUDA: configure both engines for CPU execution
if that is the tournament environment. Use the actual tournament clocks;
120+0.5 in the final example comes from agent.py comments, not verified rules.
For development throughput use --workers 3 (or 6 if memory/CPU permit) without
CPU pins. Each game creates fresh engines and pays import/JIT startup outside
the game clock. Default startup timeout is 180s; use --startup-timeout to tune.
Fresh processes prevent game history, imported modules and TT contamination.

Experiment definitions:
A1: removes missing-TT-move depth reduction.
A2: A1 + LMP only outside PV/check, through depth 5, with larger move counts.
A3: A2 + capture SEE pruning only outside PV/check, no promotions, depth <=4,
    threshold -100*depth instead of -60*depth. Qsearch SEE is unchanged.
A4: A3 + requires TWO previous keys for repetition, excludes history across
    null moves. Twofold search-cycle scoring is a common heuristic, not itself
    proof of a bug. This tests strict threefold in negamax; qsearch is unchanged.
    Supplied board.zobrist hashes any stored en-passant square, not just legal
    en-passant rights. A4 does not change this separate board-level limitation.
A5: A4 + forward futility through depth 4 with larger margins. Reverse futility
    and razoring are unchanged. These thresholds are hypotheses, not proven gains.
A6/A7: instructions only; not runnable variants and no claimed strength benefit.
If a step loses, build a new explicitly named combination from the winning
predecessor; do not silently reinterpret these cumulative A labels.

Outputs and interpretation:
PGNs are plain strings before leaving the game coordinator. The parent alone
writes them in completion order, flushing and fsyncing every finished game.
JSONL records results, errors, measured move time and counts. Summary JSON
records configuration, W/D/L, excluded games, score fraction and relative Elo.
Existing PGN/JSONL files are refused; choose new names for reruns. No resume.
Both sides flag before increment; illegal moves/crashes on move are forfeits.
Setup failures and max-plies games remain '*' and are excluded, never losses
or invented draws. Inspect exclusions/forfeits before interpreting scores.
Claimable draws are auto-claimed by the referee. Opening moves are unclocked;
agent history begins at the supplied opening endpoint, as with a FEN platform.
Hung agent moves are terminated at their remaining clock. Stockfish protocol
timeouts are bounded, then treated as forfeits. Ctrl+C preserves flushed games;
active coordinators may take their bounded game time to finish during shutdown.

Elo is relative to this matchup, not a rating or calibrated Stockfish Elo.
No confidence interval is claimed for correlated paired/repeated openings.
The ten included lines are a smoke/screening suite, not enough diversity for
a final strength claim. Supply a larger fixed opening set via --openings FILE
(one legal UCI sequence per line), retain color pairs, and repeat close results.
If benchmarks disagree, investigate opponent/opening dependence and sampling
uncertainty; neither matchup is automatically authoritative. The runners do
not compute blunders, +2 conversion rates, or extract reliable NPS/depth from
book/fallback moves; those require a separate analysis pipeline.

Validation scope: syntax plus five mock-engine/process/serialization/repetition
tests, including a concurrent CLI run with PGN/JSONL readback. Full NNUE engine
import, Numba compilation and playing strength have not been validated here.
No real-match result is claimed by this package.

Your supplied HalfKA folder can be used directly for the HalfKA smoke test:
  python play_match_elo_5600g_3060_fixed.py --engine-a "C:\Users\punam\coding\bullet_agent\pranaya_agent\trueAgentV5H" --stockfish "%SF%" --games 2 --workers 1 --base 10 --increment 0.5 --pgn-out results/local_halfka_smoke.pgn
That nnue.py prefers halfka_raw.npz when present, otherwise it loads
halfka_quantized_raw_8bit.npz. If neither exists it silently uses the fallback
evaluator. Verify the expected weight file before treating a result as NNUE.
