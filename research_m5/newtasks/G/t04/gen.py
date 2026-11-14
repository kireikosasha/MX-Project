#!/usr/bin/env python3
"""Kinds-and-properties reading comprehension generator.

Usage: python gen.py N SEED   (prints N stories to standard output)

World: individuals (names) each belong to a kind. Kinds are linked by
"Every K1 is a K2." (K1 is a subkind of K2). Kinds and individuals carry
properties. Membership and properties are inherited along subkind links.
Questions are answered from the statements above them only, with a
closed-world rule: a kind or property that cannot be derived is "no".
"""
import random
import sys
from collections import defaultdict

NAMES = ["ann", "bob", "cal", "dee", "eli", "fay", "gus", "hal"]
KINDS = ["wolf", "dog", "cat", "creature", "thing", "pet"]
ADJS = ["fast", "slow", "loud", "quiet", "furry", "happy", "tall", "small"]


def cap(word):
    return word[0].upper() + word[1:]


def render(st):
    t = st[0]
    if t == "mem":
        return f"{cap(st[1])} is a {st[2]}."
    if t == "sub":
        return f"Every {st[1]} is a {st[2]}."
    if t == "kprop":
        return f"Every {st[1]} is {st[2]}."
    return f"{cap(st[1])} is {st[2]}."  # iprop


def derive(stmts):
    """Return (kinds(name), props(name)) computed from the given statements."""
    mem = defaultdict(set)
    sup = defaultdict(set)
    kp = defaultdict(set)
    ip = defaultdict(set)
    for st in stmts:
        t = st[0]
        if t == "mem":
            mem[st[1]].add(st[2])
        elif t == "sub":
            sup[st[1]].add(st[2])
        elif t == "kprop":
            kp[st[1]].add(st[2])
        else:
            ip[st[1]].add(st[2])

    def ancestors(k):
        seen = {k}
        stack = [k]
        while stack:
            for s in sup.get(stack.pop(), ()):
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        return seen

    def kinds(name):
        out = set()
        for k in mem.get(name, ()):
            out |= ancestors(k)
        return out

    def props(name):
        out = set(ip.get(name, ()))
        for k in kinds(name):
            out |= kp.get(k, set())
        return out

    return kinds, props


def make_question(rng, stmts, introduced):
    kinds, props = derive(stmts)
    qtype = rng.choice(["prop", "kind", "who", "many"])
    adj = rng.choice(ADJS)
    if qtype == "who":
        matches = sorted(n for n in introduced if adj in props(n))
        if matches:
            return f"Who is {adj}?\t{','.join(matches)}"
        qtype = "prop"  # nobody has the property: ask a yes/no question instead
    if qtype == "many":
        count = sum(1 for n in introduced if adj in props(n))
        return f"How many are {adj}?\t{count}"
    if qtype == "prop":
        n = rng.choice(introduced)
        ans = "yes" if adj in props(n) else "no"
        return f"Is {cap(n)} {adj}?\t{ans}"
    n = rng.choice(introduced)
    k = rng.choice(KINDS)
    ans = "yes" if k in kinds(n) else "no"
    return f"Is {cap(n)} a {k}?\t{ans}"


def make_story(rng):
    names = rng.sample(NAMES, rng.randint(2, 4))
    chain = rng.sample(KINDS, 3)  # chain[0] < chain[1] < chain[2]
    mems = [("mem", nm, rng.choice(chain)) for nm in names]
    subs = rng.sample(
        [("sub", chain[0], chain[1]), ("sub", chain[1], chain[2])],
        rng.randint(1, 2),
    )
    kprops = [
        ("kprop", k, a)
        for k, a in rng.sample([(k, a) for k in chain for a in ADJS], rng.randint(1, 3))
    ]
    iprops = [
        ("iprop", n, a)
        for n, a in rng.sample([(n, a) for n in names for a in ADJS], rng.randint(0, 3))
    ]

    pending = list(mems)  # each name must be introduced before its other facts
    others = subs + kprops + iprops
    stmts, lines, introduced = [], [], []
    q_left = rng.randint(1, 5)

    while pending or others:
        eligible = [o for o in others if o[0] != "iprop" or o[1] in introduced]
        if pending and (not eligible or rng.random() < 0.5):
            st = pending.pop(0)
            introduced.append(st[1])
        else:
            st = rng.choice(eligible)
            others.remove(st)
        stmts.append(st)
        lines.append(render(st))
        if q_left and introduced and rng.random() < 0.4:
            lines.append(make_question(rng, stmts, introduced))
            q_left -= 1

    while q_left:
        lines.append(make_question(rng, stmts, introduced))
        q_left -= 1

    return [f"{i} {line}" for i, line in enumerate(lines, 1)]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
