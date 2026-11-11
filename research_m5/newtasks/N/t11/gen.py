#!/usr/bin/env python3
"""Synthetic record-log reasoning task: rules given as records, facts about
individuals, and yes/no queries that need rules applied to facts.

Usage: python gen.py N SEED [SIZE]      SIZE in {1, 3, 10}, default 1

Record rows ("<number> <op> <arg> <arg>."):
  gib a c.   individual a has property c (fact)
  lup a c.   individual a loses property c (retracts a gib fact)
  vok c d.   everything with property c has property d (rule)
  dro c d.   the rule "everything with c has d" is withdrawn

Query rows ("<number> hsa a c?<TAB>yes|no"):
  does individual a have property c, given the records above it?
  Answer is "yes" iff c is derived for a by forward chaining over the
  facts and rules currently in force.
"""
import random
import sys

INDIVIDUALS = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]
PROPERTIES = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
QUERY_OP = "hsa"
MAX_HOPS = {1: 2, 3: 3, 10: 5}
SIZES = (1, 3, 10)


def closure(facts, rules, ind):
    """Properties that individual `ind` has under forward chaining."""
    have = {c for (a, c) in facts if a == ind}
    changed = True
    while changed:
        changed = False
        for c, d in sorted(rules):
            if c in have and d not in have:
                have.add(d)
                changed = True
    return have


def random_record(rng, inds, props, facts, rules):
    """A random record valid in the current state, or None if none exists."""
    free_facts = [(a, c) for a in inds for c in props if (a, c) not in facts]
    free_rules = [(c, d) for c in props for d in props
                  if c != d and (c, d) not in rules]
    kinds, weights = [], []
    if free_facts:
        kinds.append("gib"); weights.append(3)
    if free_rules:
        kinds.append("vok"); weights.append(2)
    if facts:
        kinds.append("lup"); weights.append(1)
    if rules:
        kinds.append("dro"); weights.append(1)
    if not kinds:
        return None
    kind = rng.choices(kinds, weights)[0]
    if kind == "gib":
        return ("gib",) + rng.choice(free_facts)
    if kind == "vok":
        return ("vok",) + rng.choice(free_rules)
    if kind == "lup":
        return ("lup",) + rng.choice(sorted(facts))
    return ("dro",) + rng.choice(sorted(rules))


def apply_record(rec, facts, rules):
    op, x, y = rec
    if op == "gib":
        facts.add((x, y))
    elif op == "lup":
        facts.discard((x, y))
    elif op == "vok":
        rules.add((x, y))
    else:
        rules.discard((x, y))


def pick_query(rng, inds, props, facts, rules):
    """Choose (individual, property) and compute its yes/no answer."""
    cl = {a: closure(facts, rules, a) for a in inds}
    derived = [(a, c) for a in inds for c in props
               if c in cl[a] and (a, c) not in facts]
    absent = [(a, c) for a in inds for c in props if c not in cl[a]]
    if derived and (rng.random() < 0.5 or not absent):
        a, c = rng.choice(derived)
    elif absent:
        a, c = rng.choice(absent)
    else:
        a, c = rng.choice(sorted(facts))
    return a, c, ("yes" if c in cl[a] else "no")


def make_episode(rng, size):
    """Return the rows of one episode (without line numbers)."""
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    inds = rng.sample(INDIVIDUALS, rng.randint(2, 5))
    props = rng.sample(PROPERTIES, rng.randint(4, 8))

    # Planned rule chains c0 -> c1 -> ... -> ck, each with a fact on its start.
    planned = []
    for _ in range(rng.randint(1, 1 + size // 2 + 1)):
        hops = rng.randint(1, min(MAX_HOPS[size], len(props) - 1))
        seq = rng.sample(props, hops + 1)
        planned.append(("gib", rng.choice(inds), seq[0]))
        for c, d in zip(seq, seq[1:]):
            planned.append(("vok", c, d))
    rng.shuffle(planned)

    facts, rules = set(), set()
    records = []
    states = []  # state after each record, used for queries
    tries = 0
    while len(records) < n_rec and tries < 1000:
        tries += 1
        rec = None
        if planned and rng.random() < 0.7:
            op, x, y = planned.pop(0)
            if (x, y) not in (facts if op == "gib" else rules):
                rec = (op, x, y)
        if rec is None:
            rec = random_record(rng, inds, props, facts, rules)
        if rec is None:
            break
        apply_record(rec, facts, rules)
        records.append(rec)
        states.append((frozenset(facts), frozenset(rules)))

    slots = sorted(rng.randint(1, len(records)) for _ in range(n_q))
    queries = {}
    for slot in slots:
        facts_s, rules_s = states[slot - 1]
        queries.setdefault(slot, []).append(
            pick_query(rng, inds, props, facts_s, rules_s))

    rows = []
    for i, rec in enumerate(records, 1):
        rows.append("%s %s %s." % rec)
        for a, c, ans in queries.get(i, []):
            rows.append("%s %s %s?\t%s" % (QUERY_OP, a, c, ans))
    return rows


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
    if size not in SIZES:
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, row in enumerate(make_episode(rng, size), 1):
            out.append("%d %s" % (i, row))
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
