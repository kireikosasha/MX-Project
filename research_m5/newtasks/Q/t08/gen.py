import random
import sys

# Places lie in compass directions from each other. Every statement says that
# one place lies in a direction from another. Questions are asked only when the
# statements above them already settle the answer (checked by transitive closure
# along each axis). Answers come from a hidden simulated layout.

PLACES = ["mill", "farm", "harbor", "tower", "orchard", "quarry", "meadow"]
DIRS = ["north", "south", "east", "west"]
AXIS = {"north": 1, "south": 1, "east": 0, "west": 0}  # 0 = x, 1 = y
SIGN = {"north": 1, "south": -1, "east": 1, "west": -1}


def statement(a, b, d, rng):
    # a lies in direction d from b
    t = rng.randrange(3)
    if t == 0:
        return f"The {a} lies {d} from the {b}."
    if t == 1:
        return f"Going {d} from the {b} brings you to the {a}."
    return f"From the {b} the {a} is to the {d}."


def closure(edges):
    rel = set(edges)
    changed = True
    while changed:
        changed = False
        for p, q in list(rel):
            for r, s in list(rel):
                if q == r and (p, s) not in rel:
                    rel.add((p, s))
                    changed = True
    return rel


def build(prefix):
    raw = {0: set(), 1: set()}
    for a, b, d in prefix:
        if SIGN[d] > 0:
            raw[AXIS[d]].add((b, a))  # b < a on that axis
        else:
            raw[AXIS[d]].add((a, b))  # a < b on that axis
    return {ax: closure(raw[ax]) for ax in raw}


def known(rel, a, b, d):
    # True / False when the statements settle whether a lies in direction d from b
    r = rel[AXIS[d]]
    lo, hi = (b, a) if SIGN[d] > 0 else (a, b)
    if (lo, hi) in r:
        return True
    if (hi, lo) in r:
        return False
    return None


def make_story(rng):
    k = rng.randint(3, 6)
    names = rng.sample(PLACES, k)
    xs = list(range(k))
    rng.shuffle(xs)
    ys = list(range(k))
    rng.shuffle(ys)
    pos = {nm: (xs[i], ys[i]) for i, nm in enumerate(names)}

    def truth(a, b, d):
        i = AXIS[d]
        return SIGN[d] * (pos[a][i] - pos[b][i]) > 0

    all_facts = [(a, b, d) for a in names for b in names if a != b
                 for d in DIRS if truth(a, b, d)]
    rng.shuffle(all_facts)
    facts = all_facts[:rng.randint(3, 12)]
    n_q = rng.randint(1, 5)

    def question(prefix, mentioned, asked):
        rel = build(prefix)
        cands = []
        for a in mentioned:
            for b in mentioned:
                if a == b:
                    continue
                for d in DIRS:
                    kn = known(rel, a, b, d)
                    if kn is None:
                        continue
                    assert kn == truth(a, b, d)
                    ans = "yes" if kn else "no"
                    text = rng.choice([
                        f"Does the {a} lie {d} from the {b}",
                        f"Would walking {d} from the {b} reach the {a}",
                    ])
                    cands.append((text, ans))
            for d in DIRS:
                others = [p for p in mentioned if p != a]
                if not others or any(known(rel, p, a, d) is None for p in others):
                    continue
                hits = sorted(p for p in others if truth(p, a, d))
                if hits:
                    assert hits == sorted(p for p in others if known(rel, p, a, d))
                    cands.append((f"Which places lie {d} from the {a}", ",".join(hits)))
        cands = [c for c in cands if c[0] not in asked]
        if not cands:
            return None
        return rng.choice(cands)

    lines = []
    prefix = []
    mentioned = []
    asked = set()
    qleft = n_q
    ln = 0
    nq_done = 0
    for idx, (a, b, d) in enumerate(facts):
        prefix.append((a, b, d))
        ln += 1
        lines.append(f"{ln} {statement(a, b, d, rng)}")
        for p in (a, b):
            if p not in mentioned:
                mentioned.append(p)
        last = idx == len(facts) - 1
        while qleft > 0 and (last or rng.random() < 0.4):
            got = question(prefix, mentioned, asked)
            if got is None:
                break
            text, ans = got
            asked.add(text)
            qleft -= 1
            nq_done += 1
            ln += 1
            lines.append(f"{ln} {text}?\t{ans}")
    if nq_done == 0:
        return None
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = make_story(rng)
        out.extend(story)
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
