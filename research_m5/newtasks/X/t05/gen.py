#!/usr/bin/env python3
"""Synthetic record-log reasoning task: who knows which document.

An episode is a log of records (mk = a person makes a document, tl = a person
tells another person about a document, fg = a person forgets a document),
followed by queries about what the log implies about knowledge.

Usage: python gen.py N SEED SIZE   (SIZE is 1, 3 or 10; default 1)
"""
import random
import sys

PEOPLE = ["p1", "p2", "p3", "p4", "p5", "p6"]
DOCS = ["d1", "d2", "d3", "d4"]


def make_record(rng, knows, mentions, made):
    fresh = [d for d in DOCS if d not in made]
    tellable = [d for d in made if knows[d] and len(knows[d]) < len(PEOPLE)]
    forgettable = [d for d in made if knows[d]]
    ops = (["mk"] * (2 if fresh else 0)
           + ["tl"] * (4 if tellable else 0)
           + ["fg"] * (1 if forgettable else 0))
    if not ops:
        return None
    op = rng.choice(ops)
    if op == "mk":
        d = rng.choice(fresh)
        p = rng.choice(PEOPLE)
        made.append(d)
        knows[d].add(p)
        text = f"mk {p} {d}."
    elif op == "tl":
        d = rng.choice(tellable)
        p = rng.choice(sorted(knows[d]))
        q = rng.choice([x for x in PEOPLE if x not in knows[d]])
        knows[d].add(q)
        text = f"tl {p} {q} {d}."
    else:
        d = rng.choice(forgettable)
        p = rng.choice(sorted(knows[d]))
        knows[d].discard(p)
        text = f"fg {p} {d}."
    mentions[d] += 1
    return text


def make_query(rng, knows, mentions, made):
    # Only ask about documents that at least two records have touched,
    # so the answer has to be assembled from several records.
    cands = [d for d in made if mentions[d] >= 2]
    if not cands:
        return None
    d = rng.choice(cands)
    kind = rng.choice(["kn", "kn", "wo", "ct"])
    if kind == "kn":
        want_yes = rng.random() < 0.5
        pool = (sorted(knows[d]) if want_yes
                else [p for p in PEOPLE if p not in knows[d]])
        if not pool:
            return None
        p = rng.choice(pool)
        ans = "yes" if p in knows[d] else "no"
        return f"kn {p} {d}?\t{ans}"
    if kind == "wo":
        if not knows[d]:
            return None
        return f"wo {d}?\t" + ",".join(sorted(knows[d]))
    return f"ct {d}?\t{len(knows[d])}"


def sim_episode(rng, size):
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    knows = {d: set() for d in DOCS}
    mentions = {d: 0 for d in DOCS}
    made = []
    rows = []
    recs = 0
    qs = 0
    while recs < n_rec or qs < n_q:
        if qs < n_q and (recs >= n_rec or rng.random() < 0.25):
            q = make_query(rng, knows, mentions, made)
            if q is not None:
                rows.append(q)
                qs += 1
                continue
            if recs >= n_rec:
                return None  # no valid query left; retry the episode
        r = make_record(rng, knows, mentions, made)
        if r is None:
            return None
        rows.append(r)
        recs += 1
    return rows


def generate(n, seed, size):
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        rows = sim_episode(rng, size)
        if rows is None:
            continue
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(rows)
    return episodes


def main(argv):
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    for rows in generate(n, seed, size):
        for i, row in enumerate(rows, 1):
            print(f"{i} {row}")


if __name__ == "__main__":
    main(sys.argv)
