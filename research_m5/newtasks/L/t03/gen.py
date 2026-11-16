"""Synthetic reading-comprehension task: comparison on a scale (height).

Stories state pairwise comparisons ("x is taller than y"). Questions need
several comparisons chained by transitivity; each question is only asked
when its answer is fully determined by the statements above it, and only
when no single statement above it already gives the answer.

Usage: python gen.py N SEED
"""

import random
import sys

NAMES = ["ann", "bob", "cara", "dan", "eve", "finn", "gus", "hana", "ivy", "jack"]
QUESTION_KINDS = ["yesno", "tallest", "shortest", "taller", "shorter", "count"]


def closure(pairs):
    """Transitive closure of facts (x, y) meaning 'x is taller than y'."""
    rel = set(pairs)
    changed = True
    while changed:
        changed = False
        for a, b in list(rel):
            for c, d in list(rel):
                if b == c and a != d and (a, d) not in rel:
                    rel.add((a, d))
                    changed = True
    return rel


def listed(rel, a, universe, taller):
    """Names in universe known to be taller than a (taller=True) or shorter
    than a (taller=False), sorted. None if some name is undecided."""
    out = []
    for x in universe:
        if x == a:
            continue
        yes_pair, no_pair = ((x, a), (a, x)) if taller else ((a, x), (x, a))
        if yes_pair in rel:
            out.append(x)
        elif no_pair not in rel:
            return None
    return sorted(out)


def answer(kind, args, pairs, universe):
    """Answer determined by the facts in pairs, or None if not determined."""
    rel = closure(pairs)
    if kind == "yesno":
        a, b = args
        if (a, b) in rel:
            return "yes"
        if (b, a) in rel:
            return "no"
        return None
    if kind in ("tallest", "shortest"):
        for m in args:
            others = [x for x in args if x != m]
            if kind == "tallest":
                ok = all((m, x) in rel for x in others)
            else:
                ok = all((x, m) in rel for x in others)
            if ok:
                return m
        return None
    a = args[0]
    taller = kind in ("taller", "count")
    found = listed(rel, a, universe, taller)
    if not found:
        return None
    if kind == "count":
        return str(len(found))
    return ",".join(found)


def make_question(rng, pairs):
    """Random question about the people mentioned so far, or None."""
    universe = sorted({n for p in pairs for n in p})
    kind = rng.choice(QUESTION_KINDS)
    if kind == "yesno":
        if len(universe) < 2:
            return None
        args = tuple(rng.sample(universe, 2))
        text = "is %s taller than %s?" % args
    elif kind in ("tallest", "shortest"):
        if len(universe) < 3:
            return None
        args = tuple(rng.sample(universe, 3))
        text = "who is the %s of %s and %s and %s?" % ((kind,) + args)
    else:
        args = (rng.choice(universe),)
        text = {
            "taller": "who is taller than %s?",
            "shorter": "who is shorter than %s?",
            "count": "how many are taller than %s?",
        }[kind] % args[0]

    ans = answer(kind, args, pairs, universe)
    if ans is None:
        return None
    # The answer must need several statements: no single statement may suffice.
    for i in range(len(pairs)):
        if answer(kind, args, pairs[i:i + 1], universe) == ans:
            return None
    return text, ans


def make_story(rng):
    """Return a list of (text, answer_or_None) lines for one story."""
    while True:
        k = rng.randint(3, 6)
        world = rng.sample(NAMES, k)
        ranks = list(range(k))
        rng.shuffle(ranks)
        height = dict(zip(world, ranks))

        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        cuts = sorted(rng.randint(2, n_stmt) for _ in range(n_q))

        lines, pairs, asked = [], [], set()
        qi, ok = 0, True
        for i in range(n_stmt + 1):
            while ok and qi < n_q and cuts[qi] == i:
                found = None
                for _ in range(60):
                    q = make_question(rng, pairs)
                    if q is not None and q[0] not in asked:
                        found = q
                        break
                if found is None:
                    ok = False
                    break
                asked.add(found[0])
                lines.append(found)
                qi += 1
            if not ok or i == n_stmt:
                break
            a, b = rng.sample(world, 2)
            if height[a] < height[b]:
                a, b = b, a  # a is taller than b
            pairs.append((a, b))
            if rng.random() < 0.5:
                lines.append(("%s is taller than %s." % (a, b), None))
            else:
                lines.append(("%s is shorter than %s." % (b, a), None))
        if ok and qi == n_q:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(2)
    n_stories = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n_stories):
        for idx, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                out.append("%d %s" % (idx, text))
            else:
                out.append("%d %s\t%s" % (idx, text, ans))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
