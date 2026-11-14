"""Synthetic map-direction reading task.

Places sit on a hidden 4x4 map. Stories state, for pairs of places on the same
row or column, which way one lies from the other. Questions ask whether one
place lies in a direction from another, or which direction it lies in. Every
question is about a relation that follows only by chaining two or more
statements (no single statement gives it directly), and answers are computed
from the statements so far, checked against the hidden map.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["park", "school", "library", "market", "station", "harbor", "farm",
         "bridge", "lake", "tower", "museum", "castle", "garden", "office"]
DIRS = ["north", "south", "east", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
GRID = 4


def direction_between(p, q):
    """Direction of place p from place q on the hidden map."""
    px, py = p
    qx, qy = q
    if px == qx and py > qy:
        return "north"
    if px == qx and py < qy:
        return "south"
    if py == qy and px > qx:
        return "east"
    if py == qy and px < qx:
        return "west"
    return None


def reachable(edges, start, goal):
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for x, y in edges:
            if x == cur and y not in seen:
                if y == goal:
                    return True
                seen.add(y)
                stack.append(y)
    return False


def derived(edges, a, b, d):
    """'a is d of b' follows by chaining, and no single statement says it."""
    if (a, b) in edges[d]:
        return False
    return reachable(edges[d], a, b)


def candidate_questions(edges, names, pos):
    out = []
    for a in names:
        for b in names:
            if a == b:
                continue
            truth = direction_between(pos[a], pos[b])
            found = [d for d in DIRS if derived(edges, a, b, d)]
            if len(found) == 1:
                assert found[0] == truth
                out.append((f"Which direction is the {a} from the {b}?", found[0]))
            for d in DIRS:
                if derived(edges, a, b, d):
                    ans = "yes"
                elif derived(edges, a, b, OPP[d]):
                    ans = "no"
                else:
                    continue
                assert (truth == d) == (ans == "yes")
                out.append((f"Is the {a} {d} of the {b}?", ans))
    return out


def try_story(rng):
    count = rng.randint(4, 6)
    names = rng.sample(NAMES, count)
    cells = [(x, y) for x in range(GRID) for y in range(GRID)]
    rng.shuffle(cells)
    pos = {n: cells[i] for i, n in enumerate(names)}
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    edges = {d: set() for d in DIRS}
    used_pairs = set()
    asked = set()
    items = []  # (text, answer or None)
    stmts = 0
    qs = 0
    while stmts < n_stmt or qs < n_q:
        if qs < n_q and stmts >= 1 and (stmts == n_stmt or rng.random() < 0.4):
            options = [o for o in candidate_questions(edges, names, pos)
                       if o[0] not in asked]
            if options:
                text, ans = rng.choice(options)
                asked.add(text)
                items.append((text, ans))
                qs += 1
                continue
            if stmts == n_stmt:
                return None
        if stmts >= n_stmt:
            return None
        pairs = [(a, b) for a in names for b in names
                 if a != b
                 and direction_between(pos[a], pos[b]) is not None
                 and frozenset((a, b)) not in used_pairs]
        if not pairs:
            return None
        a, b = rng.choice(pairs)
        d = direction_between(pos[a], pos[b])
        used_pairs.add(frozenset((a, b)))
        edges[d].add((a, b))
        edges[OPP[d]].add((b, a))
        if rng.random() < 0.5:
            text = f"The {a} is {d} of the {b}."
        else:
            text = f"The {b} is {OPP[d]} of the {a}."
        items.append((text, None))
        stmts += 1
    return items


def make_story(rng):
    while True:
        items = try_story(rng)
        if items is not None:
            return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                lines.append(f"{i} {text}")
            else:
                lines.append(f"{i} {text}\t{ans}")
    if lines:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
