"""Rebuild cumulative variants from the supplied, unmodified search reference."""
from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parent


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    source = (ROOT / 'reference/search.py').read_text(encoding='utf-8')
    previous = source
    variants = [('A0', source)]
    source = replace_once(source,
        '    # with no hint from the table, a shallower search is cheaper than a badly ordered one\n    if tt_move == 0 and depth >= 5:\n        depth -= 1\n',
        '    # A1: keep requested depth even when there is no TT move.\n')
    variants.append(('A1', source))
    source = replace_once(source, 'if depth <= 7 and legal >= LATE_MOVE_COUNT[depth]:',
                          'if not pv_node and not checked and depth <= 5 and legal >= LATE_MOVE_COUNT[depth]:')
    source = replace_once(source, '[0, 6, 9, 14, 21, 30, 41, 54]', '[0, 8, 12, 20, 30, 42, 56, 72]')
    variants.append(('A2', source))
    source = replace_once(source,
        'elif capture and depth <= 6 and see(bbs, mbs, mem, ply, move) < -60 * depth:',
        'elif (capture and (move >> 12) & 7 == 0 and not checked and not pv_node\n                  and depth <= 4 and see(bbs, mbs, mem, ply, move) < -100 * depth):')
    variants.append(('A3', source))
    start = source.index('    """Has this exact position occurred')
    end = source.index('\n\n\n@njit', start)
    source = source[:start] + '''    """Require two prior occurrences; never count across a synthetic null move.

    This is a strict threefold experiment, not a claim that twofold search-cycle
    detection is always wrong. Root/game history is supplied by agent.py.
    """
    key = sts[ply, 4]
    limit = sts[ply, 3]
    base = OFF_PATH + mem[OFF_CTL + CTL_BASE]
    index = base + ply
    floor = OFF_PATH
    null_floor = mem[OFF_CTL + CTL_REP_FLOOR]
    if null_floor > 0:
        floor = base + null_floor
    matches = 0
    back = 2
    while back <= limit and index - back >= floor:
        if mem[index - back] == key:
            matches += 1
            if matches >= 2:
                return True
        back += 2
    return False''' + source[end:]
    source = replace_once(source, 'CTL_STABLE = 11', 'CTL_STABLE = 11\nCTL_REP_FLOOR = 12  # boundary after the latest synthetic null move')
    source = replace_once(source, '            make_null(bbs, sts, mbs, ply)',
        '            saved_floor = mem[OFF_CTL + CTL_REP_FLOOR]\n            mem[OFF_CTL + CTL_REP_FLOOR] = ply + 1\n            make_null(bbs, sts, mbs, ply)')
    source = replace_once(source,
        '                -beta, -beta + 1, null_off, no_move,\n            )',
        '                -beta, -beta + 1, null_off, no_move,\n            )\n            mem[OFF_CTL + CTL_REP_FLOOR] = saved_floor')
    source = replace_once(source, '    mem[OFF_CTL + CTL_NODES] = 0',
        '    mem[OFF_CTL + CTL_NODES] = 0\n    mem[OFF_CTL + CTL_REP_FLOOR] = 0')
    variants.append(('A4', source))
    source = replace_once(source, '[0, 110, 210, 320, 450, 600, 780, 980]',
                          '[0, 160, 300, 450, 620, 800, 1000, 1250]')
    source = replace_once(source, 'not pv_node and depth <= 7 and static + FUTILITY',
                          'not pv_node and depth <= 4 and static + FUTILITY')
    variants.append(('A5', source))
    source = replace_once(source, '        best_score = score\n        settled =',
        '        previous_score = best_score\n        best_score = score\n        settled =')
    source = replace_once(source, '        if not settled and depth >= 5:',
        '        score_drop = (abs(previous_score) < MATE_IN_MAX and abs(score) < MATE_IN_MAX\n                      and previous_score - score >= 80)\n        if depth >= 5 and (not settled or score_drop):')
    variants.append(('A6', source))
    start = source.index('@njit("i8(i8[:,::1]', source.index('def update_correction'))
    end = source.index('\n\n@njit(', start + 1)
    source = source[:start] + (ROOT / 'qsearch_a7.txt').read_text(encoding='utf-8').rstrip() + source[end:]
    source = replace_once(source, 'return quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply, alpha, beta)',
        'return quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply, alpha, beta, np.int64(1))')
    source = replace_once(source, 'scouted = quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply, alpha - 1, alpha)',
        'scouted = quiesce(bbs, sts, mbs, acc, psqt, mem, tt, ply, alpha - 1, alpha, np.int64(1))')
    variants.append(('A7', source))
    for name, text in variants:
        dest = ROOT / 'variants' / name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'search.py').write_text(text, encoding='utf-8')
        if name != 'A0':
            (dest / 'changes.diff').write_text(''.join(difflib.unified_diff(
                previous.splitlines(True), text.splitlines(True), fromfile='previous/search.py',
                tofile=f'{name}/search.py')), encoding='utf-8')
        previous = text


if __name__ == '__main__':
    main()
