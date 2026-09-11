"""Compile and smoke-test A6/A7 in isolated copies. Usage: --engine-dir PATH."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def child(variant):
    import time
    import chess
    import agent
    import search as s
    agent._stop_ponder()
    agent._start_ponder = lambda: None
    assert s.nnue_mod.NNUE_AVAILABLE, 'NNUE weights required for this test'
    # Exercise the actual soft-deadline decision without waiting on a clock.
    import ast
    tree = ast.parse(Path(s.__file__).read_text())
    think = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'think')
    loop = next(n for n in think.body if isinstance(n, ast.For) and getattr(n.target, 'id', '') == 'depth')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], 'id', '') == 'limit')
    decision = compile(ast.Module(body=loop.body[start:start+3], type_ignores=[]), '<deadline>', 'exec')
    for previous, score, settled, depth, expected in (
            (100, 100, True, 5, 100), (100, 20, True, 5, 200),
            (100, 21, True, 5, 100), (100, 20, False, 5, 200),
            (100, 20, True, 4, 100), (s.MATE, 20, True, 5, 100)):
        ns = dict(mem={0: 100, 1: 400}, OFF_CTL=0, CTL_SOFT_NS=0,
                  CTL_HARD_NS=1, MATE_IN_MAX=s.MATE_IN_MAX,
                  previous_score=previous, score=score, settled=settled, depth=depth)
        exec(decision, ns)
        assert ns['limit'] == expected and ns['mem'][1] == 400, 'soft deadline regression'
    for fen in (chess.STARTING_FEN, '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'):
        move = agent.get_move(fen, 3000)
        assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves
    if variant == 'A7':
        def q(fen, budget):
            board = chess.Board(fen)
            agent._load(board)
            agent._load_history([int(agent.STS[0, 4])])
            agent.MEM[s.OFF_CTL + s.CTL_STOP] = 0
            agent.MEM[s.OFF_CTL + s.CTL_REP_FLOOR] = 0
            agent.MEM[s.OFF_CTL + s.CTL_HARD_NS] = int((time.perf_counter()+10)*1e9)
            s.nnue_mod.nnue_init_root(agent.BBS[0], agent.ACC, agent.PSQT)
            result = s.quiesce(agent.BBS, agent.STS, agent.MBS, agent.ACC, agent.PSQT,
                              agent.MEM, agent.TT, 0, -s.INF, s.INF, budget)
            assert not agent.MEM[s.OFF_CTL+s.CTL_STOP], 'qsearch timed out'
            return result
        mate = '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
        assert q(mate, 1) > s.MATE_IN_MAX, 'quiet-check mate missed'
        assert q(mate, 0) < s.MATE_IN_MAX, 'quiet-check budget not respected'
        assert q('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1', 1) == 0, 'stalemate'
        assert q('7k/6Q1/6K1/8/8/8/8/8 b - - 100 1', 0) == -s.MATE, 'mate precedes draw'
        assert q('r6k/8/8/8/8/8/8/K7 w - - 0 1', 0) > -s.MATE_IN_MAX, 'mandatory evasion'
    print(f'{variant}: NNUE compiled, legal moves and runtime checks PASS', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine-dir', type=Path)
    p.add_argument('--child', choices=['A6', 'A7'])
    args = p.parse_args()
    if args.child:
        child(args.child)
        return
    if args.engine_dir is None:
        p.error('--engine-dir is required')
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        for name in ('agent.py', 'board.py', 'position.py', 'evaluate.py', 'magics.py', 'nnue.py'):
            shutil.copy2(args.engine_dir/name, folder/name)
        for name in ('halfka_raw.npz', 'halfka_quantized_raw_8bit.npz'):
            if (args.engine_dir/name).exists():
                shutil.copy2(args.engine_dir/name, folder/name)
                break
        shutil.copy2(__file__, folder/'test_search_runtime.py')
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
        for variant in ('A6', 'A7'):
            shutil.copy2(root/'variants'/variant/'search.py', folder/'search.py')
            subprocess.run([sys.executable, str(folder/'test_search_runtime.py'), '--child', variant],
                           cwd=folder, env=env, timeout=240, check=True)


if __name__ == '__main__':
    main()
