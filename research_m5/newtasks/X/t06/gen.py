#!/usr/bin/env python3
"""Synthetic record task: pairwise links between entities are added, removed and
cleared by records; queries ask about the link state at that point.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)
"""
import random
import sys

REC_OPS = ("tev", "vulp", "kipu")  # link two entities / unlink two / clear all links of one
QRY_OPS = ("fyz", "nax", "kot")    # linked? / list of neighbours / number of links
KINDS = ("c", "p")                 # one letter per kind of entity
DIGITS = range(1, 9)               # at most 8 codes per kind


def pool(kind):
    return [f"{kind}{d}" for d in DIGITS]


class Links:
    def __init__(self):
        self.pairs = set()  # set of frozensets of two entity codes

    def linked(self, a, b):
        return frozenset((a, b)) in self.pairs

    def neighbours(self, a):
        return sorted(next(iter(p - {a})) for p in self.pairs if a in p)

    def apply(self, rec):
        op = rec[0]
        if op == "tev":
            self.pairs.add(frozenset((rec[1], rec[2])))
        elif op == "vulp":
            self.pairs.discard(frozenset((rec[1], rec[2])))
        elif op == "kipu":
            self.pairs = {p for p in self.pairs if rec[1] not in p}


def make_record(rng, links):
    r = rng.random()
    if r < 0.6:
        a, b = rng.sample(pool(rng.choice(KINDS)), 2)
        return ("tev", a, b)
    if r < 0.85:
        existing = sorted(tuple(sorted(p)) for p in links.pairs)  # sorted: reproducible
        if existing and rng.random() < 0.7:
            a, b = rng.choice(existing)
        else:
            a, b = rng.sample(pool(rng.choice(KINDS)), 2)
        return ("vulp", a, b)
    return ("kipu", rng.choice(pool(rng.choice(KINDS))))


def make_query(rng, links, hist):
    """Return (query_text, answer) or None. Only asks about entities that appear
    in at least two earlier records, so the answer needs several records."""
    touch = {}
    for ents in hist:
        for e in ents:
            touch[e] = touch.get(e, 0) + 1
    ents = sorted(touch)
    if len(ents) < 2:
        return None
    for _ in range(60):
        op = rng.choice(QRY_OPS)
        if op == "fyz":
            kind = rng.choice(KINDS)
            same = [e for e in ents if e[0] == kind]
            if len(same) < 2:
                continue
            existing = sorted(tuple(sorted(p)) for p in links.pairs if p and next(iter(p))[0] == kind)
            if existing and rng.random() < 0.5:
                a, b = rng.choice(existing)
            else:
                a, b = rng.sample(same, 2)
            n_touch = sum(1 for ents_r in hist if ents_r & {a, b})
            if n_touch < 2:
                continue
            ans = "yes" if links.linked(a, b) else "no"
            return f"fyz {a} {b}?", ans
        a = rng.choice(ents)
        if touch[a] < 2:
            continue
        if op == "kot":
            return f"kot {a}?", str(len(links.neighbours(a)))
        nb = links.neighbours(a)
        if nb:
            return f"nax {a}?", ",".join(nb)
    return None


def episode_try(rng, size):
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    plan = ["R"] * n_rec + ["Q"] * n_q
    rng.shuffle(plan)
    if plan[0] == "Q":  # the first row must be a record
        i = plan.index("R")
        plan[0], plan[i] = plan[i], plan[0]
    links = Links()
    hist = []
    rows = []
    for item in plan:
        if item == "Q":
            q = make_query(rng, links, hist)
            if q is None:
                return None
            rows.append(q[0] + "\t" + q[1])
        else:
            rec = make_record(rng, links)
            links.apply(rec)
            hist.append(set(rec[1:]))
            rows.append(" ".join(rec) + ".")
    return rows


def episode(rng, size):
    for _ in range(10000):
        rows = episode_try(rng, size)
        if rows is not None:
            return rows
    raise RuntimeError("could not build an episode")


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    for _ in range(n):
        for i, row in enumerate(episode(rng, size), 1):
            print(f"{i} {row}")


if __name__ == "__main__":
    main(sys.argv)
