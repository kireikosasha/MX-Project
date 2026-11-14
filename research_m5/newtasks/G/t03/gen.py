"""Synthetic task: comparison on a scale.

Each story has a few people with hidden heights, weights and ages. Statements
compare pairs of people on one scale ("ann is taller than bob."). Questions ask
whether one person is taller/heavier/older than another, or who is the
tallest/shortest/heaviest/lightest/oldest/youngest of a small group. A question
is only asked when the statements above it already fix its answer through a
chain of comparisons (transitive closure), and the answer is read from the
simulated world.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "bob", "carl", "dana", "eli", "fay", "gus", "hana", "ivo", "jade"]

# (scale name, comparative high, comparative low, superlative high, superlative low)
ATTRS = [
    ("height", "taller", "shorter", "tallest", "shortest"),
    ("weight", "heavier", "lighter", "heaviest", "lightest"),
    ("age", "older", "younger", "oldest", "youngest"),
]
ATTR_BY_NAME = {a[0]: a for a in ATTRS}


def reach(edges, src, dst):
    """True if src is above dst on one scale, via a chain of stated comparisons.

    edges[u] is the set of people stated to be directly below u.
    """
    stack = [src]
    seen = {src}
    while stack:
        u = stack.pop()
        for v in edges.get(u, ()):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def make_statement(rng, names, world, edges, direct):
    # Prefer a pair whose order is not yet known; otherwise repeat a known fact.
    cands = []
    for attr in ATTRS:
        an = attr[0]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if not reach(edges[an], a, b) and not reach(edges[an], b, a):
                    cands.append((an, a, b))
    if cands:
        an, a, b = rng.choice(cands)
    else:
        an = rng.choice(ATTRS)[0]
        a, b = rng.sample(names, 2)
    if world[an][a] < world[an][b]:
        a, b = b, a  # a is the higher one now
    edges[an].setdefault(a, set()).add(b)
    direct[an].add((a, b))
    attr = ATTR_BY_NAME[an]
    if rng.random() < 0.5:
        return f"{a} is {attr[1]} than {b}."
    return f"{b} is {attr[2]} than {a}."


def make_question(rng, names, world, edges, direct):
    """Return (question_text, answer) if the statements above fix the answer
    through a chain of at least two comparisons; otherwise None."""
    for _ in range(40):
        attr = rng.choice(ATTRS)
        an = attr[0]
        vals = world[an]
        if rng.random() < 0.5:
            # "Is a taller than b?" -> yes/no
            a, b = rng.sample(names, 2)
            hi, lo = (a, b) if vals[a] > vals[b] else (b, a)
            if (hi, lo) in direct[an] or (lo, hi) in direct[an]:
                continue  # answered by a single sentence, not a chain
            if not reach(edges[an], hi, lo):
                continue  # not yet determined
            ans = "yes" if vals[a] > vals[b] else "no"
            return f"Is {a} {attr[1]} than {b}?", ans
        # "Who is the tallest of a and b and c?" -> one name
        group = rng.sample(names, rng.choice([2, 3]))
        want_max = rng.random() < 0.5
        x = max(group, key=lambda n: vals[n]) if want_max else min(group, key=lambda n: vals[n])
        ok = True
        for y in group:
            if y == x:
                continue
            hi, lo = (x, y) if want_max else (y, x)
            if not reach(edges[an], hi, lo):
                ok = False
                break
            if len(group) == 2 and ((hi, lo) in direct[an] or (lo, hi) in direct[an]):
                ok = False  # single sentence decides it
                break
        if not ok:
            continue
        sup = attr[3] if want_max else attr[4]
        return f"Who is the {sup} of {' and '.join(group)}?", x
    return None


def try_story(rng):
    k = rng.randint(3, 5)
    names = rng.sample(NAMES, k)
    world = {a[0]: dict(zip(names, rng.sample(range(1, 100), k))) for a in ATTRS}
    edges = {a[0]: {} for a in ATTRS}
    direct = {a[0]: set() for a in ATTRS}
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    stmts_left, q_left, stmts_done = n_stmt, n_q, 0
    lines = []  # (text, answer or None)
    while stmts_left or q_left:
        if q_left and stmts_done >= 2 and (not stmts_left or rng.random() < 0.35):
            q = make_question(rng, names, world, edges, direct)
            if q is not None:
                lines.append(q)
                q_left -= 1
                continue
            if not stmts_left:
                return None
        if not stmts_left:
            return None
        lines.append((make_statement(rng, names, world, edges, direct), None))
        stmts_left -= 1
        stmts_done += 1
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                print(f"{i} {text}")
            else:
                print(f"{i} {text}\t{ans}")


if __name__ == "__main__":
    main()
