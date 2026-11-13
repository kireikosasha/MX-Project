import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dev", "Eli"]
THINGS = ["apple", "coin"]


def tw(n, thing):
    return thing if n == 1 else thing + "s"


def transaction(rng, names, counts):
    kind = rng.choice(["gets", "gives", "takes", "loses"])
    thing = rng.choice(THINGS)
    known = [n for n in names if (n, thing) in counts]
    if kind == "loses":
        cands = [n for n in known if counts[(n, thing)] >= 1]
        if not cands:
            return None
        x = rng.choice(cands)
        k = rng.randint(1, min(3, counts[(x, thing)]))
        counts[(x, thing)] -= k
        return f"{x} loses {k} {tw(k, thing)}.", [(x, thing)]
    givers = [n for n in known if counts[(n, thing)] >= 1]
    if not givers or len(known) < 2:
        return None
    a = rng.choice(givers)
    b = rng.choice([n for n in known if n != a])
    k = rng.randint(1, min(3, counts[(a, thing)]))
    counts[(a, thing)] -= k
    counts[(b, thing)] += k
    if kind == "gives":
        text = f"{a} gives {k} {tw(k, thing)} to {b}."
    else:
        verb = "gets" if kind == "gets" else "takes"
        text = f"{b} {verb} {k} {tw(k, thing)} from {a}."
    return text, [(a, thing), (b, thing)]


def intro(rng, counts, unknown):
    n, t = rng.choice(unknown)
    c = rng.randint(1, 6)
    counts[(n, t)] = c
    return f"{n} has {c} {tw(c, t)}.", [(n, t)], False


def statement(rng, names, counts):
    unknown = [(n, t) for n in names for t in THINGS if (n, t) not in counts]
    if unknown and rng.random() < 0.35:
        return intro(rng, counts, unknown)
    for _ in range(20):
        r = transaction(rng, names, counts)
        if r is not None:
            return r[0], r[1], True
    if unknown:
        return intro(rng, counts, unknown)
    return None


def question(rng, names, counts, hist, is_tx):
    for _ in range(30):
        kind = rng.choice(["how", "most", "more"])
        thing = rng.choice(THINGS)
        pool = [n for n in names if (n, thing) in counts]
        if kind == "how":
            if not pool:
                continue
            x = rng.choice(pool)
            text = f"How many {thing}s does {x} have?"
            ans = str(counts[(x, thing)])
            rel = [(x, thing)]
        elif kind == "most":
            if len(pool) < 2:
                continue
            top = max(counts[(n, thing)] for n in pool)
            winners = [n for n in pool if counts[(n, thing)] == top]
            if top == 0 or len(winners) != 1:
                continue
            text = f"Who has the most {thing}s?"
            ans = winners[0].lower()
            rel = [(n, thing) for n in pool]
        else:
            if len(pool) < 2:
                continue
            x, y = rng.sample(pool, 2)
            if counts[(x, thing)] == counts[(y, thing)]:
                continue
            text = f"Does {x} have more {thing}s than {y}?"
            ans = "yes" if counts[(x, thing)] > counts[(y, thing)] else "no"
            rel = [(x, thing), (y, thing)]
        support = set()
        for p in rel:
            support |= hist.get(p, set())
        if len(support) >= 2 and any(is_tx[i] for i in support):
            return text, ans
    return None


def attempt(rng):
    names = rng.sample(NAMES, rng.randint(2, 4))
    S = rng.randint(3, 12)
    Q = rng.randint(1, 5)
    counts, hist, is_tx, lines = {}, {}, [], []
    qs = 0
    while len(is_tx) < S or qs < Q:
        if qs < Q and len(is_tx) >= 2 and (len(is_tx) >= S or rng.random() < 0.35):
            q = question(rng, names, counts, hist, is_tx)
            if q is not None:
                lines.append(q)
                qs += 1
                continue
            if len(is_tx) >= S:
                return None
        st = statement(rng, names, counts)
        if st is None:
            return None
        text, pairs, tx = st
        idx = len(is_tx)
        is_tx.append(tx)
        for p in pairs:
            hist.setdefault(p, set()).add(idx)
        lines.append((text, None))
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lines = None
        while lines is None:
            lines = attempt(rng)
        for num, (text, ans) in enumerate(lines, 1):
            if ans is None:
                out.append(f"{num} {text}")
            else:
                out.append(f"{num} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
