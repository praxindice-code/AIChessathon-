import ast
import pickle
from pathlib import Path
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import json
import unittest

import chess
import chess.pgn
from benchmark_common import play_game, score_result


class BenchmarkTests(unittest.TestCase):
    def test_concurrent_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'agent.py').write_text('import chess\ndef _stop_ponder(): pass\ndef _start_ponder(): pass\ndef get_move(fen, ms): return next(iter(chess.Board(fen).legal_moves)).uci()\n')
            output = root/'games.pgn'
            command = [sys.executable, str(Path(__file__).parent/'play_match_head_to_head.py'),
                       '--engine-a', temp, '--engine-b', temp, '--games', '4', '--workers', '2',
                       '--max-plies', '2', '--pgn-out', str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            rows = [json.loads(line) for line in output.with_suffix('.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual({row['index'] for row in rows}, set(range(4)))
            with output.open() as stream:
                games = []
                while (game := chess.pgn.read_game(stream)) is not None:
                    games.append(game)
            self.assertEqual(len(games), 4)
            self.assertEqual(games[0].headers['Result'], '*')

    def test_score(self):
        self.assertIsNone(score_result('*', True))
        self.assertEqual(score_result('1-0', False), 0)
        self.assertEqual(score_result('1/2-1/2', True), 0.5)

    def test_long_pgn(self):
        board = chess.Board()
        for _ in range(400):
            for move in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
                board.push_uci(move)
        text = chess.pgn.Game.from_board(board).accept(chess.pgn.StringExporter())
        self.assertEqual(pickle.loads(pickle.dumps({'pgn': text}))['pgn'], text)

    def test_process_games(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, expression in [('legal', 'next(iter(chess.Board(fen).legal_moves)).uci()'),
                                      ('illegal', '"a1a8"'), ('hang', '__import__("time").sleep(10)')]:
                folder = root / name
                folder.mkdir()
                (folder / 'agent.py').write_text(
                    'import chess\ndef _stop_ponder(): pass\ndef _start_ponder(): pass\n'
                    f'def get_move(fen, ms): return {expression}\n')
            args = SimpleNamespace(engine_a=str(root/'legal'), engine_b=str(root/'legal'),
                cpu_a=None, cpu_b=None, ponder=False, increment=0.5, base=1,
                startup_timeout=10, stockfish=None, max_plies=2)
            result = play_game(0, '', args)
            self.assertIsNone(result['score'])
            self.assertEqual(result['moves'], [1, 1])
            args.engine_b = str(root/'illegal')
            result = play_game(0, '', args)
            self.assertEqual(result['score'], 1)
            self.assertIn('ValueError', result['termination'])
            args.engine_b = str(root/'hang')
            args.base = 0.2
            result = play_game(0, '', args)
            self.assertEqual(result['score'], 1)
            self.assertIn('TimeoutError', result['termination'])

    def test_repetition(self):
        # Run the actual function body without importing the unavailable engine.
        tree = ast.parse((Path(__file__).parent/'variants/A4/search.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'repeated')
        fn.decorator_list = []
        ns = dict(OFF_PATH=32, OFF_CTL=0, CTL_BASE=5, CTL_REP_FLOOR=12)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<repeated>', 'exec'), ns)
        mem = [0] * 100
        mem[5] = 4
        mem[32] = mem[34] = 99
        sts = {(0, 4): 99, (0, 3): 4}
        self.assertTrue(ns['repeated'](sts, mem, 0))
        mem[32] = 77
        self.assertFalse(ns['repeated'](sts, mem, 0))
        mem[32] = 99
        mem[12] = 1
        self.assertFalse(ns['repeated'](sts, mem, 0))


if __name__ == '__main__':
    unittest.main()
