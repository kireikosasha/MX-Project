"""One-link lexicon: the meaning of a sentence kind is read off the stories in which it stands alone; the solver finds
the machine over the few read symbols; the kinds never seen alone (or whose one-link stories disagree) are read by the
machine found (the reading that explains most of the stories they stand in), or by the solver over the alphabet of the
read symbols plus those raw ones.

General prior used: "a one-word story says what the word means": the answer of the stories that consist of one link
(view, sign) is taken as a symbol naming that link, when those stories agree (a clear majority, >= 2/3). The answer
words are only used as opaque names of the symbols; nothing reads their meaning.

    python run.py --shape learn        (the shape found too: cells.learn over the small alphabet, pool mode)
    python run.py --shape register     (the shape fixed to the register shape: cells.fit)
    add --unknown solver  to read the unknown kinds with the solver (cells.fit_greedy over read + raw symbols)
"""
import argparse, os, pickle, random, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv, ARGV = ['x'], sys.argv
sys.path.insert(0, '../..')
import cells  # noqa: E402

FOLDER = 'StepGame/Dataset/TrainVersion'
REGISTER = (0, 1, (("Z", 1, 1), ("Z", 1, 1)), True, False)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load():
    path = os.path.join(HERE, 'data.pkl')
    if not os.path.exists(path):
        import torch
        torch.set_num_threads(1)
        import one
        taught, tried = one.stepgame(FOLDER)

        def seq_of(s):
            st = one.steps_of(s)
            return None if st is None else tuple(st) + (("asked", s[6]),)
        teach = [(seq_of(s), s[4], s[5]) for s in taught]
        test = {}
        for name, some in tried.items():  # 1000 test stories per k, drawn with random.Random(1); used only at the end
            test[name] = [(seq_of(s), s[4], s[5]) for s in random.Random(1).sample(some, 1000)]
        pickle.dump({'teach': teach, 'test': test}, open(path, 'wb'))
    return pickle.load(open(path, 'rb'))


def greedy(shape, syms, labels, data, part, seconds, rng, batch=10):
    """cells.fit_greedy, begun from a given part (stories every symbol of which is read: they agree): of the stories
    the machine answers wrongly, `batch` are added; if no machine explains them all, one by one; a story that cannot
    be added is set aside. -> the machine, the stories set aside"""
    aside = []
    m, how = cells.fit(shape, syms, labels, part, seconds)
    if m is None:
        return None, how
    while True:
        wrong = [x for x in data if x not in aside and x not in part and m.run(x[0]) != x[1]]
        log(f"  {len(part)} stories explained, {len(aside)} set aside, {len(wrong)} still answered wrongly")
        if not wrong:
            return m, aside
        more = rng.sample(wrong, min(batch, len(wrong)))
        got, how = cells.fit(shape, syms, labels, part + more, seconds)
        if got is not None:
            m, part = got, part + more
            continue
        for x in more:
            got, how = cells.fit(shape, syms, labels, part + [x], seconds)
            if got is not None:
                m, part = got, part + [x]
            else:
                aside.append(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', choices=['learn', 'register'], default='learn')
    ap.add_argument('--unknown', choices=['machine', 'solver'], default='machine')
    ap.add_argument('--agree', type=float, default=2 / 3, help='share of one-link stories that must agree')
    ap.add_argument('--start', type=int, default=60, help='read stories taught on before asking (learn)')
    ap.add_argument('--asks', type=int, default=40)
    ap.add_argument('--seconds', type=float, default=10)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per', type=int, default=10, help='--unknown solver: stories per unknown kind')
    a = ap.parse_args(ARGV[1:])
    d = load()
    # ---- different teaching stories: grouped by their symbols, the answer given most often
    by = defaultdict(Counter)
    for s, y, _ in d['teach']:
        if s is not None:
            by[s][y] += 1
    diff = {s: c.most_common(1)[0][0] for s, c in by.items()}
    labels = sorted({y for y in diff.values()})
    raw_syms = sorted({x for s in diff for x in s}, key=repr)
    log(f"{len(diff)} different teaching stories, {len(raw_syms)} symbols, {len(labels)} answers")
    # ---- the reading from one-link stories: (view, sign) -> the answer of the stories where it stands alone
    alone = defaultdict(Counter)
    for s, c in by.items():
        if len(s) == 2:
            alone[s[0]].update(c)
    read = {}
    for v, c in alone.items():
        y, n = c.most_common(1)[0]
        if n >= a.agree * sum(c.values()):
            read[v] = "r:" + y
    asks = sorted({s[-1] for s in diff}, key=repr)
    for i, q in enumerate(asks):
        read[q] = f"q{i}"
    small = sorted(set(read.values()))
    unknown = [x for x in raw_syms if x not in read]
    log(f"read from one-link stories: {len(read) - len(asks)} kinds into {len(small) - len(asks)} symbols "
        f"(+{len(asks)} question); {len(alone) - (len(read) - len(asks))} seen alone but disagreeing, "
        f"{len(unknown)} left unknown")
    # ---- the read stories: every symbol read; grouped again (several raw stories -> one read story), weighted majority
    rby = defaultdict(Counter)
    for s, c in by.items():
        if all(x in read for x in s):
            rby[tuple(read[x] for x in s)].update(c)
    rdata = sorted(((s, c.most_common(1)[0][0]) for s, c in rby.items()), key=repr)
    conflicting = sum(len(c) > 1 for c in rby.values())
    log(f"{sum(all(x in read for x in s) for s in diff)} different stories fully read -> {len(rdata)} read stories "
        f"({conflicting} with answers that disagree)")
    rdict = dict(rdata)
    rng = random.Random(a.seed)
    # ---- the machine over the small alphabet, found by the solver
    asked = 0
    t1 = time.time()
    if a.shape == 'learn':
        data = rng.sample(rdata, a.start)
        told = []
        oracle = lambda w: (told.append((tuple(w), rdict[tuple(w)])), rdict[tuple(w)])[1]
        pool = [s for s, _ in rdata]
        for rnd in range(10):
            # learn asks from the pool (150 drawn each turn); when it stops, the machine kept is checked on every read
            # story taught on, and up to three it answers wrongly are asked as well (the check by trying, from the pool)
            m, _ = cells.learn(small, oracle, data, rng, asks=a.asks, seconds=a.seconds, longest=6, pool=pool,
                               workers=a.workers)
            if m is None:
                log("learn found nothing"); return
            data = data + [x for x in told if x not in data]
            told.clear()
            wrong = [x for x in rdata if m.run(x[0]) != x[1]]
            log(f"learn kept {m} ({m.shape[0]:.0f} bits); {len(data)} stories ({a.start} taught, {len(data) - a.start} "
                f"asked); wrong on {len(wrong)} of {len(rdata)} read stories ({time.time() - t1:.1f} s)")
            if not wrong:
                break
            data = data + rng.sample(wrong, min(3, len(wrong)))
        asked = len(data) - a.start
        shape = m.shape
    else:
        shape = REGISTER
        m, how = cells.fit(shape, small, labels, rdata, 300)
        log(f"register shape on all {len(rdata)} read stories: {how} ({time.time() - t1:.1f} s)")
        if m is None:
            return
    print(m.table(), flush=True)
    # ---- the unknown kinds
    t2 = time.time()
    if a.unknown == 'machine':
        # each unknown kind is read as the machine symbol that explains most of the different stories it stands in
        # (others read or already settled); repeated until nothing changes
        res = {}
        for turn in range(5):
            changed = 0
            for u in unknown:
                stories = [(s, y) for s, y in diff.items() if u in s and all(x in read or x in res or x == u for x in s)]
                if not stories:
                    continue
                score = Counter()
                for c in small:
                    look = dict(read); look.update(res); look[u] = c
                    score[c] = sum(m.run(tuple(look[x] for x in s)) == y for s, y in stories)
                best = max(score.values())
                top = sorted(c for c in small if score[c] == best)[0]
                if res.get(u) != top:
                    res[u], changed = top, changed + 1
            if not changed:
                break
        lexicon = dict(read); lexicon.update(res)
        machine = m
        log(f"{len(res)} unknown kinds read by the machine ({time.time() - t2:.1f} s)")
        run = lambda s: None if s is None or any(x not in lexicon for x in s) else machine.run(tuple(lexicon[x] for x in s))
    else:
        # the solver: the same shape over the read symbols plus the unknown raw kinds, on every different story
        # (read symbols mapped), stories that cannot be explained set aside (cells.fit_greedy)
        ext = small + unknown
        mby = defaultdict(Counter)
        for s, c in by.items():
            mby[tuple(read.get(x, x) for x in s)].update(c)
        mdata = sorted(((s, c.most_common(1)[0][0]) for s, c in mby.items()), key=repr)
        # the stories: every read story and, of every unknown kind, at most --per of the stories it stands in;
        # begun from 200 read stories
        part = list(rdata)
        for u in unknown:
            has = [x for x in mdata if u in x[0]]
            part += rng.sample(has, min(a.per, len(has)))
        part = list(dict.fromkeys(part))
        log(f"solver over {len(ext)} symbols on {len(part)} stories")
        machine, aside = greedy(shape, ext, labels, part, rng.sample(rdata, 200), 60, rng)
        log(f"solver over {len(ext)} symbols: {len(aside)} set aside ({time.time() - t2:.1f} s)")
        asked = (asked, len(aside))
        known = set(ext)
        run = lambda s: None if s is None or any(read.get(x, x) not in known for x in s) else machine.run(tuple(read.get(x, x) for x in s))
    # ---- measure
    right = sum(run(s) == y for s, y in diff.items())
    log(f"RESULT taught: {right}/{len(diff)} = {100 * right / len(diff):.2f}% of the different teaching stories answered rightly")
    wt = sum(run(s) == y for s, y, _ in d['teach'])
    log(f"RESULT taught (all 50000, with repeats): {100 * wt / len(d['teach']):.2f}%")
    accs, nones = [], []
    for name, some in d['test'].items():
        got = [run(s) for s, _, _ in some]
        accs.append(100 * sum(g == y for g, (_, y, _) in zip(got, some)) / len(some))
        nones.append(100 * sum(g is None for g in got) / len(some))
    log("RESULT test % by k=1..10: " + " ".join(f"{x:.1f}" for x in accs))
    log("RESULT no answer % by k=1..10: " + " ".join(f"{x:.1f}" for x in nones))
    log(f"RESULT wall {time.time() - T0:.1f} s; asked {asked}")


if __name__ == '__main__':
    main()
