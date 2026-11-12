#!/usr/bin/env python3
"""Assignment by elimination, told as records.

Hidden rule of every episode: a one-to-one assignment of people (p-codes) to
spots (s-codes). Every person takes exactly one spot and every spot is taken
by exactly one person.

Record rows:
  asg p s.   person p takes spot s
  nox p s.   person p cannot take spot s

Query rows (answered only from the records above them; the answer is computed
by enumerating every assignment that fits those records):
  xat p?     the one spot p can take                 -> one s-code
  xls p?     all spots p can still take, sorted      -> s-codes joined by commas
  xok p s?   can p take s?                           -> yes / no

Usage: python gen.py N SEED SIZE    (SIZE is 1, 3 or 10; default 1)
"""

import random
import sys

PEOPLE = 8
SPOTS = 8
MAX_PEOPLE = {1: 4, 3: 6, 10: 8}


def matchings(k, recs):
    """All full assignments (person index -> spot index) that fit the records."""
    allowed = [set(range(k)) for _ in range(k)]
    for op, i, j in recs:
        if op == "asg":
            allowed[i] &= {j}
        else:
            allowed[i].discard(j)
    result = []

    def extend(i, taken, cur):
        if i == k:
            result.append(tuple(cur))
            return
        for j in sorted(allowed[i]):
            if not ((taken >> j) & 1):
                cur.append(j)
                extend(i + 1, taken | (1 << j), cur)
                cur.pop()

    extend(0, 0, [])
    return result


def candidates(k, recs, P, S):
    """Queries that the records above are enough to answer: (text, answer)."""
    ms = matchings(k, recs)
    possible = [sorted({m[i] for m in ms}) for i in range(k)]
    cnt_p = [0] * k
    cnt_s = [0] * k
    direct = [False] * k
    has_nox = [False] * k
    pairs = set()
    for op, i, j in recs:
        cnt_p[i] += 1
        cnt_s[j] += 1
        pairs.add((i, j))
        if op == "asg":
            direct[i] = True
        else:
            has_nox[i] = True
    if min(cnt_p) == 0 or min(cnt_s) == 0:
        # every person and spot must already be named above the query,
        # so the reader knows the full set of candidates
        return []
    out = []
    for i in range(k):
        p = P[i]
        opts = possible[i]
        if len(opts) == 1 and not direct[i] and cnt_p[i] >= 2:
            out.append(("xat %s?" % p, S[opts[0]]))
        if 2 <= len(opts) < k and has_nox[i] and not direct[i]:
            out.append(("xls %s?" % p, ",".join(sorted(S[j] for j in opts))))
        if cnt_p[i] >= 1:
            for j in range(k):
                if (i, j) not in pairs and cnt_s[j] >= 1:
                    out.append(("xok %s %s?" % (p, S[j]),
                                "yes" if j in opts else "no"))
    return out


def gen_episode(rng, size):
    k = rng.randint(3, MAX_PEOPLE[size])
    P = ["p%d" % n for n in rng.sample(range(1, PEOPLE + 1), k)]
    S = ["s%d" % n for n in rng.sample(range(1, SPOTS + 1), k)]
    hidden = list(range(k))
    rng.shuffle(hidden)  # hidden[i] = spot index taken by person i
    pool = []
    for i in range(k):
        for j in range(k):
            pool.append(("asg" if j == hidden[i] else "nox", i, j))
    target = rng.randint(3, 12) * (1 if size == 1 else size)
    recs = rng.sample(pool, min(target, len(pool)))
    want = rng.randint(1, 5)
    rows = []
    placed = 0
    used = set()
    for n in range(1, len(recs) + 1):
        op, i, j = recs[n - 1]
        rows.append("%s %s %s." % (op, P[i], S[j]))
        if placed < want and rng.random() < 0.6:
            options = [c for c in candidates(k, recs[:n], P, S)
                       if c[0] not in used]
            if options:
                kinds = sorted({c[0].split()[0] for c in options})
                kind = rng.choice(kinds)
                q, a = rng.choice([c for c in options
                                   if c[0].startswith(kind + " ")])
                used.add(q)
                rows.append(q + "\t" + a)
                placed += 1
    if placed == 0:
        return None
    return rows


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in MAX_PEOPLE:
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    out = []
    count = 0
    while count < n:
        rows = gen_episode(rng, size)
        if rows is None or tuple(rows) in seen:
            continue
        seen.add(tuple(rows))
        for num, row in enumerate(rows, 1):
            out.append("%d %s" % (num, row))
        count += 1
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv)
