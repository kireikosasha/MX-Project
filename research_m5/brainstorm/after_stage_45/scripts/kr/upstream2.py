"""The upstream cell enumerated exactly (every assignment of shifts/resets of Z_m to the symbols), the downstream counter
fitted per (symbol, upstream state) by exact coordinate descent; the machine kept by fit, then carry-over, then fewest states."""
import itertools, random, time, sys
import numpy as np
sys.argv += ["80"]
import importlib.util
spec = importlib.util.spec_from_file_location("t", "turtle.py"); 
src = open("turtle.py").read().split("def main():")[0]
exec(src)
OFFA = np.array(OFF)

def fit_counter(counts, gold_cls, cells):
    """counts [S, cells]; gold class 0..8 = 3*sx+sy -> offsets index per cell, score (readout by majority)"""
    o = np.zeros(cells, dtype=int) + OFF.index((0, 0))
    def score(o):
        pos = counts @ OFFA[o]
        cls = 3 * (np.sign(pos[:, 0]) + 1) + (np.sign(pos[:, 1]) + 1)
        tb = np.zeros((9, 9), dtype=int); np.add.at(tb, (cls, gold_cls), 1)
        return tb.max(1).sum(), tb.argmax(1)
    best = score(o)[0]
    for _ in range(5):
        moved = False
        for c in range(cells):
            for j in range(9):
                if j == o[c]: continue
                was = o[c]; o[c] = j; sc = score(o)[0]
                if sc > best: best, moved = sc, True
                else: o[c] = was
        if not moved: break
    return o, best, score(o)[1]

def machine(train, m, aside):
    sid = {x: i for i, x in enumerate(SYM)}
    hmaps = [[(s + k) % m for s in range(m)] for k in range(m)] + [[v] * m for v in range(m)]
    gold = np.array([g for _, g in train])
    best = None
    for combo in itertools.product(range(len(hmaps)), repeat=len(SYM)):
        counts = np.zeros((len(train), len(SYM) * m), dtype=int)
        for i, (s, _) in enumerate(train):
            h = 0
            for c in s:
                counts[i, sid[c] * m + h] += 1
                h = hmaps[combo[sid[c]]][h]
        o, sc, read = fit_counter(counts, gold, len(SYM) * m)
        if sc < len(train) and best is not None and best[0] == len(train):
            continue
        car = carry(combo, o, read, hmaps, sid, m, aside)
        if best is None or (sc, car) > (best[0], best[4]):
            best = (sc, combo, o.copy(), read, car)
    sc, combo, o, read, _ = best
    def run(s):
        h, x, y = 0, 0, 0
        for c in s:
            dx, dy = OFF[o[sid[c] * m + h]]; x += dx; y += dy
            h = hmaps[combo[sid[c]]][h]
        return int(read[3 * ((x > 0) - (x < 0) + 1) + ((y > 0) - (y < 0) + 1)])
    return run, sc / len(train)

def carry(combo, o, read, hmaps, sid, m, aside):
    ok = 0
    for s, g in aside:
        h, x, y = 0, 0, 0
        for c in s:
            dx, dy = OFF[o[sid[c] * m + h]]; x += dx; y += dy
            h = hmaps[combo[sid[c]]][h]
        ok += int(read[3 * ((x > 0) - (x < 0) + 1) + ((y > 0) - (y < 0) + 1)]) == g
    return ok / len(aside)

rng = random.Random(1)
per = int(sys.argv[1])
train = data(rng, per, 1, 4); aside = data(rng, 300, 5, 8)
for m in (2, 3, 4, 5):
    t0 = time.time()
    f, fit = machine(train, m, aside)
    car = sum(f(s) == g for s, g in aside) / len(aside)
    test = {Lg: sum(f(s) == g for s, g in data(rng, 500, Lg, Lg)) / 500 for Lg in (20, 100)}
    print(f"upstream Z{m} enumerated -> counter: fit {100*fit:5.1f}  carry-over(5-8) {100*car:5.1f}  len20 {100*test[20]:5.1f}  len100 {100*test[100]:5.1f}  ({time.time()-t0:.0f}s)", flush=True)
