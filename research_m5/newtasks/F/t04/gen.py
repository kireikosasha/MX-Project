"""Synthetic kinds-and-properties reading task (bAbI style, own vocabulary).

Usage: python gen.py N SEED

Facts (each kind of fact has one fixed pattern):
  X is a K.            individual X belongs to kind K
  Every K is a K2.     every K is a K2 (kind hierarchy)
  Every K is P.        every K has property P
  X is P.              individual X has property P

Questions (each kind of question has one fixed pattern):
  Is X a K?            membership, derived through the hierarchy
  Is X P?              property, derived from the individual or from its kinds
  Which kinds is X?    all kinds of X, most specific first, comma separated

Convention: a fact that cannot be derived from the statements above a
question is false (closed world), so the answer is always determined.
A "yes" is only asked when its derivation uses at least two statements.
"""

import random
import sys

NAMES = ["Sam", "Tia", "Ron", "Lee", "Max", "Ada", "Ivy", "Ned"]
KINDS = ["wolf", "canine", "cat", "feline", "bird", "fish", "animal"]
PARENT = {
    "wolf": "canine",
    "canine": "animal",
    "cat": "feline",
    "feline": "animal",
    "bird": "animal",
    "fish": "animal",
}
PROPS = ["fierce", "loud", "quiet", "swift", "gentle"]


def art(word):
    return ("an " if word[0] in "aeiou" else "a ") + word


def render(f):
    kind = f[0]
    if kind == "isa":
        return f"{f[1]} is {art(f[2])}."
    if kind == "sub":
        return f"Every {f[1]} is {art(f[2])}."
    if kind == "kprop":
        return f"Every {f[1]} is {f[2]}."
    return f"{f[1]} is {f[2]}."  # prop


def derive(prefix, x):
    """Return (kinds, props) of individual x, each mapping to the set of
    statement indices used to derive it. kinds is ordered by discovery."""
    kinds = {}
    queue = []
    for i, f in enumerate(prefix):
        if f[0] == "isa" and f[1] == x and f[2] not in kinds:
            kinds[f[2]] = frozenset([i])
            queue.append(f[2])
    qi = 0
    while qi < len(queue):
        k = queue[qi]
        qi += 1
        for j, f in enumerate(prefix):
            if f[0] == "sub" and f[1] == k and f[2] not in kinds:
                kinds[f[2]] = kinds[k] | {j}
                queue.append(f[2])
    props = {}
    for i, f in enumerate(prefix):
        if f[0] == "prop" and f[1] == x and f[2] not in props:
            props[f[2]] = frozenset([i])
    for k in queue:
        for j, f in enumerate(prefix):
            if f[0] == "kprop" and f[1] == k and f[2] not in props:
                props[f[2]] = kinds[k] | {j}
    return kinds, props


def pick_question(prefix, rng, asked):
    names = []
    for f in prefix:
        if f[0] in ("isa", "prop") and f[1] not in names:
            names.append(f[1])
    yes, no = [], []
    for x in names:
        kinds, props = derive(prefix, x)
        order = list(kinds)
        if len(order) >= 2 and len(set().union(*kinds.values())) >= 2:
            yes.append((f"Which kinds is {x}", ",".join(order)))
        for k in KINDS:
            q = f"Is {x} {art(k)}"
            if k in kinds:
                if len(kinds[k]) >= 2:
                    yes.append((q, "yes"))
            else:
                no.append((q, "no"))
        for p in PROPS:
            q = f"Is {x} {p}"
            if p in props:
                if len(props[p]) >= 2:
                    yes.append((q, "yes"))
            else:
                no.append((q, "no"))
    yes = [c for c in yes if c[0] not in asked]
    no = [c for c in no if c[0] not in asked]
    if not yes and not no:
        return None
    if yes and (not no or rng.random() < 0.75):
        q = rng.choice(yes)
    else:
        q = rng.choice(no)
    asked.add(q[0])
    return f"{q[0]}?\t{q[1]}"


def try_story(rng):
    names = rng.sample(NAMES, rng.randint(2, 3))
    kind_prop = {k: rng.choice(PROPS) for k in KINDS}
    pool = []

    def add(f):
        if f not in pool:
            pool.append(f)

    for x in names:
        add(("isa", x, rng.choice(KINDS)))
        if rng.random() < 0.3:
            add(("isa", x, rng.choice(KINDS)))
        if rng.random() < 0.35:
            add(("prop", x, rng.choice(PROPS)))
    for k, p in PARENT.items():
        add(("sub", k, p))
    for k in KINDS:
        add(("kprop", k, kind_prop[k]))

    n = min(rng.randint(3, 12), len(pool))
    statements = rng.sample(pool, n)

    lines = []
    prefix = []
    asked = set()
    nq = 0
    for f in statements:
        prefix.append(f)
        lines.append(render(f))
        if nq < 5 and rng.random() < 0.45:
            q = pick_question(prefix, rng, asked)
            if q:
                lines.append(q)
                nq += 1
    if nq == 0:
        q = pick_question(prefix, rng, asked)
        if q is None:
            return None
        lines.append(q)
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    main()
