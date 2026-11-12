#!/usr/bin/env python3
"""Synthetic record-log reasoning task: rules and facts given as coded rows.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)

Row kinds
  kol a b.     rule: every individual with property a has property b
  zib k a.     fact: individual k has property a
  qex k a?     query: does k have a?            answer yes / no
  mup a?       query: how many individuals have a   answer a number
  zor a?       query: which individuals have a      answer sorted codes, comma-separated

Answers come from a forward-chaining simulation over the rows above the query.
"""
import random
import sys

INDIVIDUALS = ["k%d" % i for i in range(1, 9)]
PROPERTIES = ["f%d" % i for i in range(1, 9)]
ALL_PAIRS = [(k, f) for k in INDIVIDUALS for f in PROPERTIES]

RULE_OP = "kol"
FACT_OP = "zib"
QA_OP = "qex"
QC_OP = "mup"
QL_OP = "zor"


def max_hops(size):
    """Longest rule chain allowed: 2 hops at size 1, longer at larger sizes."""
    return min(7, size + 1)


def rules_ok(rules, hops):
    """Rule graph must be acyclic and have no chain longer than `hops` rules."""
    succ = {p: [] for p in PROPERTIES}
    for a, b in rules:
        succ[a].append(b)
    memo = {}
    visiting = set()

    def depth(p):
        if p in memo:
            return memo[p]
        if p in visiting:
            raise ValueError("cycle")
        visiting.add(p)
        d = 0
        for q in succ[p]:
            d = max(d, 1 + depth(q))
        visiting.discard(p)
        memo[p] = d
        return d

    try:
        longest = max(depth(p) for p in PROPERTIES)
    except ValueError:
        return False
    return longest <= hops


def make_records(rng, size):
    """Return a shuffled list of ('rule', a, b) and ('fact', k, f) tuples."""
    hops = max_hops(size)
    total = rng.randint(3, 12) * size
    n_rules = rng.randint(max(1, total // 4), max(1, total // 2))
    rules = []
    tries = 0
    while len(rules) < n_rules and tries < 400:
        tries += 1
        a, b = rng.sample(PROPERTIES, 2)
        if (a, b) in rules:
            continue
        if rules_ok(rules + [(a, b)], hops):
            rules.append((a, b))
    n_facts = min(total - len(rules), len(ALL_PAIRS))
    facts = rng.sample(ALL_PAIRS, n_facts)
    recs = [("rule", a, b) for a, b in rules] + [("fact", k, f) for k, f in facts]
    rng.shuffle(recs)
    return recs


def closure(recs):
    """Forward chaining. Returns {k: {f: support}} where support is the set of
    record indices (in recs) used to derive that k has f."""
    rules = [(i, r[1], r[2]) for i, r in enumerate(recs) if r[0] == "rule"]
    known = {k: {} for k in INDIVIDUALS}
    for i, r in enumerate(recs):
        if r[0] == "fact":
            known[r[1]][r[2]] = frozenset([i])
    for k in INDIVIDUALS:
        queue = list(known[k].keys())
        qi = 0
        while qi < len(queue):
            p = queue[qi]
            qi += 1
            for ri, a, b in rules:
                if a == p and b not in known[k]:
                    known[k][b] = known[k][p] | {ri}
                    queue.append(b)
    return known


def make_query(rng, known):
    """Return (row_text_without_number, reasoned) or None if no query fits."""
    x = rng.random()
    if x < 0.5:
        derived = [(k, f) for k in INDIVIDUALS for f, sup in known[k].items() if len(sup) >= 2]
        if derived and rng.random() < 0.6:
            k, f = rng.choice(derived)
        else:
            k = rng.choice(INDIVIDUALS)
            f = rng.choice(PROPERTIES)
        if f in known[k]:
            ans = "yes"
            reasoned = len(known[k][f]) >= 2
        else:
            ans = "no"
            reasoned = False
        return "%s %s %s?\t%s" % (QA_OP, k, f, ans), reasoned
    if x < 0.7:
        f = rng.choice(PROPERTIES)
        n = sum(1 for k in INDIVIDUALS if f in known[k])
        return "%s %s?\t%d" % (QC_OP, f, n), False
    f = rng.choice(PROPERTIES)
    members = [k for k in INDIVIDUALS if f in known[k]]  # already sorted
    if not members:
        return None
    return "%s %s?\t%s" % (QL_OP, f, ",".join(members)), False


def make_episode(rng, size):
    """Return the episode as a list of numbered rows."""
    while True:
        recs = make_records(rng, size)
        nq = rng.randint(1, 5)
        spots = sorted(rng.randint(1, len(recs)) for _ in range(nq))
        queries = []
        reasoned = False
        for spot in spots:
            known = closure(recs[:spot])
            q = None
            while q is None:
                q = make_query(rng, known)
            text, ok = q
            reasoned = reasoned or ok
            queries.append((spot, text))
        if not reasoned:
            continue
        body = []
        qi = 0
        for idx in range(len(recs) + 1):
            while qi < len(queries) and queries[qi][0] == idx:
                body.append(queries[qi][1])
                qi += 1
            if idx < len(recs):
                r = recs[idx]
                if r[0] == "rule":
                    body.append("%s %s %s." % (RULE_OP, r[1], r[2]))
                else:
                    body.append("%s %s %s." % (FACT_OP, r[1], r[2]))
        return ["%d %s" % (i, row) for i, row in enumerate(body, 1)]


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        ep = make_episode(rng, size)
        key = "\n".join(ep)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(ep)
    out = []
    for ep in episodes:
        out.extend(ep)
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
