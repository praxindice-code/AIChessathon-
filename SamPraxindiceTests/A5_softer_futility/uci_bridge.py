"""Drop this into each engine variation's folder (alongside agent.py). Reads one request per
line from stdin as "<fen>|<time_left_ms>", calls this folder's own agent.get_move(), and
writes the UCI move to stdout. Runs until stdin closes.

This exists so two different weight sets (e.g. two different bullet_raw.npz checkpoints) can
run in separate, isolated Python processes -- agent.py/nnue.py load weights as module-level
globals at import time, so two different checkpoints can't coexist in a single process.
"""

import sys

import agent

sys.stdout.reconfigure(line_buffering=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    fen, time_left_ms = line.rsplit("|", 1)
    move = agent.get_move(fen, int(time_left_ms))
    print(move, flush=True)
