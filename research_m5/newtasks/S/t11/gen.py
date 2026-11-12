#!/usr/bin/env python3
"""Generate synthetic records-style reasoning episodes on standard output.

usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1)

Every row is a record or a query. Records are rules (ulk, fom) or facts
(bzi, hel, wgt) about individuals a1..a8, categories c1..c8 and properties
p1..p8. Queries (hez, mun, nuq) are answered by simulating the records above
them, so the answer needs rules to be applied to facts.
"""
import random
import sys
from collections import deque

INDIV, CATS, PROPS = "a", "c", "p"

R_SUB = "ulk"     # ulk ci cj.  every member of ci is a member of cj
R_PROP = "fom"    # fom ci pj.  every member of ci has property pj
F_MEMB = "bzi"    # bzi ai ci.  individual ai is a member of ci
F_HAS = "hel"     # hel ai pj.  individual ai has property pj
F_WEIGHT = "wgt"  # wgt ai n.   individual ai has weight n
Q_HAS = "hez"     # hez ai pj?  yes or no: does ai have pj?
Q_LIST = "mun"    # mun pj?     individuals that have pj, sorted, comma-separated
Q_SUM = "nuq"     # nuq pj?     sum of the weights of the individuals that have pj

REC_KINDS = ["memb", "sub", "prop", "has", "wgt"]
REC_WEIGHTS = [30, 25, 20, 12, 13]


def closure(recs):
    """Simulate the given records.

    Returns (has, weight):
      has[a]    maps each property p that a has to the smallest set of record
                indices that shows it (a direct fact, or membership plus rules)
      weight[a] = (value, record index) for individuals with a weight record
    """
    memb, subs, rules, direct, weight = {}, {}, {}, {}, {}
    for i, (op, x, y) in enumerate(recs):
        if op == R_SUB:
            subs.setdefault(x, []).append((y, i))
        elif op == R_PROP:
            rules.setdefault(x, []).append((y, i))
        elif op == F_MEMB:
            memb.setdefault(x, []).append((y, i))
        elif op == F_HAS:
            direct.setdefault(x, []).append((y, i))
        elif op == F_WEIGHT:
            weight[x] = (y, i)

    has = {}
    for a in set(memb) | set(direct):
        best = {}
        for p, i in direct.get(a, []):
            best[p] = frozenset([i])
        # breadth-first over categories, so the first support found is smallest
        reach = {}
        queue = deque()
        for c, i in memb.get(a, []):
            if c not in reach:
                reach[c] = frozenset([i])
                queue.append(c)
        while queue:
            c = queue.popleft()
            for p, i in rules.get(c, []):
                cand = reach[c] | {i}
                if p not in best or len(cand) < len(best[p]):
                    best[p] = cand
            for d, i in subs.get(c, []):
                if d not in reach:
                    reach[d] = reach[c] | {i}
                    queue.append(d)
        has[a] = best
    return has, weight


def render_record(rec):
    op, x, y = rec
    return f"{op} {x} {y}."


def make_query(rng, prefix, indivs, props):
    """Return a query row (with its answer) answered from prefix, or None."""
    has, weight = closure(prefix)
    kind = rng.choice(["has", "list", "sum"])

    if kind == "has":
        pairs = [(a, p) for a in indivs for p in props]
        yes = [(a, p) for a, p in pairs if p in has.get(a, {})]
        if yes and rng.random() < 0.5:
            a, p = rng.choice(yes)
        else:
            a, p = rng.choice(pairs)
        sup = has.get(a, {}).get(p)
        if sup is not None:
            if len(sup) < 2:
                return None
            ans = "yes"
        else:
            # a "no" needs a membership of a and at least one rule to rule out
            if not any(op == F_MEMB and x == a for op, x, _ in prefix):
                return None
            if not any(op in (R_SUB, R_PROP) for op, _, _ in prefix):
                return None
            ans = "no"
        return f"{Q_HAS} {a} {p}?\t{ans}"

    p = rng.choice(props)
    holders = sorted(a for a in has if p in has[a])
    if not holders:
        return None

    if kind == "list":
        support = set().union(*(has[a][p] for a in holders))
        if len(support) < 2:
            return None
        return f"{Q_LIST} {p}?\t" + ",".join(holders)

    if any(a not in weight for a in holders):
        return None
    support = set().union(*(has[a][p] for a in holders))
    support |= {weight[a][1] for a in holders}
    if len(support) < 2:
        return None
    total = sum(weight[a][0] for a in holders)
    return f"{Q_SUM} {p}?\t{total}"


def capacity(nA, nC, nP):
    """Number of distinct records the pools can make."""
    return nA * nC + nC * (nC - 1) // 2 + nC * nP + nA * nP + nA


def try_episode(rng, size):
    """Build one episode as a list of row texts, or None if it failed."""
    dims = [rng.randint(3, 8), rng.randint(3, 8), rng.randint(2, 8)]
    n_rec = rng.randint(3 * size, 12 * size)
    # larger episodes need enough distinct records; widen the pools if needed
    while capacity(*dims) < n_rec + 10:
        open_dims = [j for j in range(3) if dims[j] < 8]
        if not open_dims:
            break
        dims[min(open_dims, key=lambda j: dims[j])] += 1
    nA, nC, nP = dims

    indivs = [INDIV + str(i) for i in sorted(rng.sample(range(1, 9), nA))]
    cats = [CATS + str(i) for i in sorted(rng.sample(range(1, 9), nC))]
    props = [PROPS + str(i) for i in sorted(rng.sample(range(1, 9), nP))]
    order = rng.sample(cats, nC)  # ulk rules only point from earlier to later
    max_weight = 20 * size

    recs, seen, weighted = [], set(), set()
    for _ in range(n_rec):
        rec = None
        for _attempt in range(200):
            kind = rng.choices(REC_KINDS, weights=REC_WEIGHTS)[0]
            if kind == "memb":
                cand = (F_MEMB, rng.choice(indivs), rng.choice(cats))
            elif kind == "sub":
                i, j = sorted(rng.sample(range(nC), 2))
                cand = (R_SUB, order[i], order[j])
            elif kind == "prop":
                cand = (R_PROP, rng.choice(cats), rng.choice(props))
            elif kind == "has":
                cand = (F_HAS, rng.choice(indivs), rng.choice(props))
            else:
                free = [a for a in indivs if a not in weighted]
                if not free:
                    continue
                cand = (F_WEIGHT, rng.choice(free), rng.randint(1, max_weight))
            if cand not in seen:
                rec = cand
                break
        if rec is None:
            return None
        seen.add(rec)
        recs.append(rec)
        if rec[0] == F_WEIGHT:
            weighted.add(rec[1])

    # queries: each one is answered from the records above its position
    n_q = rng.randint(1, 5)
    queries = []
    for _ in range(n_q):
        for _attempt in range(30):
            cut = rng.randint(1, n_rec)
            text = make_query(rng, recs[:cut], indivs, props)
            if text is not None:
                queries.append((cut, text))
                break
    if not queries:
        return None

    queries.sort(key=lambda t: t[0])  # stable: keeps generation order on ties
    lines, qi = [], 0
    for idx, rec in enumerate(recs, start=1):
        lines.append(render_record(rec))
        while qi < len(queries) and queries[qi][0] == idx:
            lines.append(queries[qi][1])
            qi += 1
    return [f"{k} {text}" for k, text in enumerate(lines, start=1)]


def make_episode(rng, size, seen_keys):
    while True:
        ep = try_episode(rng, size)
        if ep is None:
            continue
        key = "\n".join(ep)
        if key not in seen_keys:
            seen_keys.add(key)
            return ep


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    try:
        n = int(argv[1])
        seed = int(argv[2])
        size = int(argv[3]) if len(argv) == 4 else 1
    except ValueError:
        sys.stderr.write("N, SEED and SIZE must be integers\n")
        return 2
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2

    rng = random.Random(seed)
    seen_keys = set()
    out = []
    for _ in range(max(n, 0)):
        out.extend(make_episode(rng, size, seen_keys))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
