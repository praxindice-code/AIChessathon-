"""Clock-based benchmark derived from reference/play_match_elo_5600g_3060.py.

Game coordinators are threads; agents have fresh spawned processes per game.
Only strings/scalars cross process boundaries. Only the parent writes results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import chess
import chess.engine
import chess.pgn


def agent_worker(conn, folder, cpu, ponder, increment):
    try:
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            os.environ[name] = '1'
        if cpu is not None:
            import psutil
            psutil.Process().cpu_affinity([cpu])
        os.chdir(folder)
        sys.path.insert(0, folder)
        agent = importlib.import_module('agent')
        if not ponder:
            if not hasattr(agent, '_start_ponder') or not hasattr(agent, '_stop_ponder'):
                raise RuntimeError('Ponder control requires the supplied agent API; adapt this bridge for this engine.')
            agent._stop_ponder()
            agent._start_ponder = lambda: None
        if hasattr(agent, 'GAME'):
            agent.GAME.increment_ms = increment * 1000
        conn.send({'ready': True})
        while True:
            request = conn.recv()
            if request is None:
                break
            move = agent.get_move(*request)
            conn.send({'move': move})
    except EOFError:
        pass
    except BaseException:
        try:
            conn.send({'error': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()


class Agent:
    def __init__(self, folder, cpu, args):
        ctx = mp.get_context('spawn')
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=agent_worker,
            args=(child, folder, cpu, args.ponder, args.increment))
        self.process.start()
        child.close()
        try:
            reply = self.receive(args.startup_timeout)
            if not reply.get('ready'):
                raise RuntimeError(reply)
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        if not self.conn.poll(max(0, timeout)):
            raise TimeoutError('agent response deadline exceeded')
        reply = self.conn.recv()
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def move(self, board, remaining):
        self.conn.send((board.fen(), max(1, int(remaining * 1000))))
        return chess.Move.from_uci(self.receive(remaining)['move'])

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(2)
        self.conn.close()


def score_result(result, a_white):
    if result == '1/2-1/2':
        return 0.5
    if result not in ('1-0', '0-1'):
        return None
    return float((result == '1-0') == a_white)


def play_game(index, opening, args):
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    a_white = index % 2 == 0
    clocks = [args.base, args.base]  # A then B
    spent = [0.0, 0.0]
    counts = [0, 0]
    agents = []
    sf = None
    reason = ''
    result = '*'
    error = None
    started = time.perf_counter()
    try:
        agents.append(Agent(args.engine_a, args.cpu_a, args))
        if args.stockfish:
            sf = chess.engine.SimpleEngine.popen_uci(args.stockfish, timeout=args.startup_timeout)
            sf.configure({'Skill Level': args.skill, 'Threads': args.sf_threads, 'Hash': args.sf_hash})
            if args.cpu_b is not None:
                import psutil
                psutil.Process(sf.transport.get_pid()).cpu_affinity([args.cpu_b])
        else:
            agents.append(Agent(args.engine_b, args.cpu_b, args))
        while not board.is_game_over(claim_draw=True):
            if sum(counts) >= args.max_plies:
                reason = 'maximum plies; unfinished, excluded from score'
                break
            side = 0 if board.turn == a_white else 1
            t0 = time.perf_counter()
            try:
                if side == 1 and sf is not None:
                    # Clock-based play uses this bounded protocol timeout;
                    # measured elapsed time still decides the clock forfeit.
                    limit = chess.engine.Limit(
                        white_clock=clocks[0 if a_white else 1],
                        black_clock=clocks[1 if a_white else 0],
                        white_inc=args.increment, black_inc=args.increment)
                    sf.timeout = max(1.0, clocks[side])
                    move = sf.play(board, limit).move
                else:
                    move = agents[side].move(board, clocks[side])
                elapsed = time.perf_counter() - t0
                spent[side] += elapsed
                counts[side] += 1
                if elapsed >= clocks[side]:
                    raise TimeoutError('clock expired before increment')
                if move not in board.legal_moves:
                    raise ValueError(f'illegal move: {move}')
            except Exception as exc:
                result = '0-1' if board.turn == chess.WHITE else '1-0'
                reason = f'{"A" if side == 0 else "B"} forfeit: {type(exc).__name__}'
                error = str(exc)
                break
            clocks[side] += args.increment - elapsed
            board.push(move)
        else:
            result = board.result(claim_draw=True)
            reason = str(board.outcome(claim_draw=True).termination)
    except Exception:
        reason = 'setup/infrastructure failure; excluded from score'
        error = traceback.format_exc()
    finally:
        for agent in agents:
            agent.close()
        if sf is not None:
            sf.close()
    game = chess.pgn.Game.from_board(board)
    b_name = f'Stockfish skill {args.skill}' if args.stockfish else Path(args.engine_b).name
    a_name = Path(args.engine_a).name
    game.headers.update({'White': a_name if a_white else b_name,
        'Black': b_name if a_white else a_name, 'Result': result,
        'Termination': reason, 'Round': str(index + 1),
        'OpeningPair': str(index // 2 + 1), 'TimeControl': f'{args.base}+{args.increment}'})
    return dict(index=index, result=result, a_white=a_white, score=score_result(result, a_white),
                termination=reason, error=error, spent=spent, moves=counts,
                elapsed=time.perf_counter() - started,
                pgn=game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)))


def main(stockfish_mode):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-a', default='.', help='folder containing agent.py and its dependencies')
    if stockfish_mode:
        p.add_argument('--stockfish', required=True)
        p.set_defaults(engine_b=None)
    else:
        p.add_argument('--engine-b', required=True)
        p.set_defaults(stockfish=None)
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--base', type=float, default=60)
    p.add_argument('--increment', type=float, default=0.5)
    p.add_argument('--skill', type=int, default=15)
    p.add_argument('--sf-threads', type=int, default=1)
    p.add_argument('--sf-hash', type=int, default=64)
    p.add_argument('--max-plies', type=int, default=600)
    p.add_argument('--startup-timeout', type=float, default=180)
    p.add_argument('--openings', default=str(Path(__file__).with_name('openings.txt')))
    p.add_argument('--pgn-out', default='elo_games.pgn')
    p.add_argument('--ponder', action='store_true', help='enable agent pondering (default off)')
    p.add_argument('--cpu-a', type=int)
    p.add_argument('--cpu-b', type=int)
    args = p.parse_args()
    if (args.games <= 0 or args.games % 2 or args.workers < 1 or args.base <= 0
            or args.increment < 0 or args.max_plies < 1 or args.startup_timeout <= 0
            or not 0 <= args.skill <= 20 or args.sf_threads < 1 or args.sf_hash < 1
            or not math.isfinite(args.base) or not math.isfinite(args.increment)):
        p.error('Use positive limits, even games, nonnegative increment, skill 0..20.')
    if (args.cpu_a is not None or args.cpu_b is not None) and args.workers != 1:
        p.error('CPU pinning requires --workers 1 to avoid pinning concurrent games together.')
    for field in ('engine_a', 'engine_b', 'stockfish'):
        value = getattr(args, field)
        if value:
            setattr(args, field, str(Path(value).resolve()))
    openings = [line.strip() for line in Path(args.openings).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    if not openings:
        p.error('Opening file is empty.')
    for line in openings:
        board = chess.Board()
        for move in line.split():
            board.push_uci(move)
        if board.is_game_over(claim_draw=True):
            p.error('Opening must be nonterminal.')
    out = Path(args.pgn_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental destruction or mixing of old runs.
    records = []
    with out.open('x', encoding='utf-8') as pgn, out.with_suffix('.jsonl').open('x', encoding='utf-8') as log:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(play_game, i, openings[(i // 2) % len(openings)], args): i
                       for i in range(args.games)}
            for future in as_completed(futures):
                try:
                    item = future.result()
                except Exception:
                    item = dict(index=futures[future], score=None, result='*', error=traceback.format_exc())
                if 'pgn' in item:
                    pgn.write(item.pop('pgn') + '\n\n')
                    pgn.flush()
                    os.fsync(pgn.fileno())
                log.write(json.dumps(item) + '\n')
                log.flush()
                os.fsync(log.fileno())
                records.append(item)
                print(f"game {item['index']+1}/{args.games}: {item['result']} {item.get('termination', '')}", flush=True)
    scores = [r['score'] for r in records if r['score'] is not None]
    wins, draws, losses = (scores.count(s) for s in (1.0, 0.5, 0.0))
    fraction = sum(scores) / len(scores) if scores else None
    elo = 400 * math.log10(fraction / (1-fraction)) if fraction and fraction < 1 else None
    summary = dict(config=vars(args), wins=wins, draws=draws, losses=losses,
                   excluded=len(records)-len(scores), score_fraction=fraction, relative_elo=elo)
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
