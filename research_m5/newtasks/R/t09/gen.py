#!/usr/bin/env python3
"""Synthetic records-only reasoning episodes: chains of one relation.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)

A record "lut c p." places entity c directly under p (p is above c). A record
"dro c." removes c's upward link. Every entity has at most one upward link at
any time, so the links always form a forest. A query asks for the entity two
steps above x ("kop x?"), the top of x's chain ("tep x?"), or whether y is
above x ("abv y x?"). Answers are simulated from the records above the query.
"""
import random
import sys

ENTS = ["k%d" % i for i in range(1, 9)]
REC_OPS = ("lut", "dro")
QRY_OPS = ("kop", "tep", "abv")


def ancestors(parent, x):
    """Entities above x, nearest first. parent maps child -> parent."""
    out = []
    while x in parent:
        x = parent[x]
        out.append(x)
    return out


def gen_records(rng, ents, n_rec):
    """Return the record rows and the state (parent map) after each row."""
    parent = {}
    rows = []
    states = [{}]
    for _ in range(n_rec):
        if parent and rng.random() < 0.25:
            c = rng.choice(sorted(parent))
            del parent[c]
            rows.append("dro %s." % c)
        else:
            pairs = []
            for c in ents:
                below = [x for x in ents if c in ancestors(parent, x)]
                for p in ents:
                    if p != c and p != parent.get(c) and p not in below:
                        pairs.append((c, p))
            if pairs:
                c, p = rng.choice(pairs)
                parent[c] = p
                rows.append("lut %s %s." % (c, p))
            else:
                c = rng.choice(sorted(parent))
                del parent[c]
                rows.append("dro %s." % c)
        states.append(dict(parent))
    return rows, states


def query_options(parent, ents):
    """(group, text, answer) for every query the state can answer non-trivially."""
    out = []
    for x in ents:
        anc = ancestors(parent, x)
        if len(anc) >= 2:
            out.append(("kop", "kop %s?" % x, anc[1]))
            out.append(("tep", "tep %s?" % x, anc[-1]))
    for x in ents:
        anc = ancestors(parent, x)
        for y in ents:
            if y != x:
                yes = y in anc
                out.append(("abv-yes" if yes else "abv-no",
                            "abv %s %s?" % (y, x),
                            "yes" if yes else "no"))
    return out


def pick_query(rng, parent, ents, used):
    groups = {}
    for grp, text, ans in query_options(parent, ents):
        if text not in used:
            groups.setdefault(grp, []).append((text, ans))
    if not groups:
        return None
    return rng.choice(groups[rng.choice(sorted(groups))])


def try_episode(rng, size):
    n_rec = rng.randint(3 * size, 12 * size)
    n_qry = rng.randint(1, 5)
    m = rng.randint(3, min(8, 4 + size // 2))
    ents = rng.sample(ENTS, m)
    recs, states = gen_records(rng, ents, n_rec)
    placed, used, attempts = [], set(), 0
    while len(placed) < n_qry:
        attempts += 1
        if attempts > 300:
            return None
        p = rng.randint(1, n_rec)  # query goes after record p
        picked = pick_query(rng, states[p], ents, used)
        if picked:
            text, ans = picked
            used.add(text)
            placed.append((p, text + "\t" + ans))
    rows = []
    for i, rec in enumerate(recs, 1):
        rows.append(rec)
        rows.extend(q for pos, q in placed if pos == i)
    return rows


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    episodes, seen = [], set()
    while len(episodes) < n:
        rows = try_episode(rng, size)
        if rows is None:
            continue
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(rows)
    blocks = ["\n".join("%d %s" % (i, r) for i, r in enumerate(rows, 1))
              for rows in episodes]
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main(sys.argv)
