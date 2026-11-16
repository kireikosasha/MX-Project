"""Synthetic place-map reasoning stories.

Places lie on a grid. Statements state that one place is one step north,
south, east or west of another. Questions ask for the overall direction
between two places (or yes/no about it), which needs chaining several
statements. Answers are computed from the statements above each question.

Usage: python gen.py N SEED
"""
import random
import sys
from collections import deque

PLACES = ["park", "lake", "farm", "mill", "barn", "hill",
          "pond", "well", "tower", "bridge", "cave", "field"]
VEC = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
VEC_TO_DIR = {v: d for d, v in VEC.items()}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
DIRS = list(VEC)


def make_blob(rng, k):
    """Random connected set of k distinct grid cells, starting at the origin."""
    cells = [(0, 0)]
    occupied = {(0, 0)}
    while len(cells) < k:
        x, y = rng.choice(cells)
        vx, vy = VEC[rng.choice(DIRS)]
        cell = (x + vx, y + vy)
        if cell not in occupied:
            occupied.add(cell)
            cells.append(cell)
    return cells


def solve(facts):
    """Relative positions implied by the facts.

    A fact (a, d, b) means pos(a) - pos(b) = VEC[d]. Returns (pos, comp):
    pos maps each place to its offset from its component root, and comp
    maps each place to its component root.
    """
    graph = {}
    for a, d, b in facts:
        vx, vy = VEC[d]
        graph.setdefault(a, []).append((b, -vx, -vy))
        graph.setdefault(b, []).append((a, vx, vy))
    pos, comp = {}, {}
    for root in graph:
        if root in pos:
            continue
        pos[root] = (0, 0)
        comp[root] = root
        queue = deque([root])
        while queue:
            u = queue.popleft()
            ux, uy = pos[u]
            for v, dx, dy in graph[u]:
                if v not in pos:
                    pos[v] = (ux + dx, uy + dy)
                    comp[v] = root
                    queue.append(v)
    return pos, comp


def direction(dx, dy):
    """Direction of a pure-axis offset (exactly one component is zero)."""
    if dx == 0:
        return "north" if dy > 0 else "south"
    return "east" if dx > 0 else "west"


def question_candidates(facts):
    """(x, y, d) where x lies in direction d from y, determined by the facts.

    Only pairs in the same component, with a pure-axis offset, and not
    stated directly, so at least two statements are needed.
    """
    pos, comp = solve(facts)
    linked = {frozenset((a, b)) for a, _, b in facts}
    names = list(pos)
    out = []
    for x in names:
        for y in names:
            if x == y or comp[x] != comp[y] or frozenset((x, y)) in linked:
                continue
            dx = pos[x][0] - pos[y][0]
            dy = pos[x][1] - pos[y][1]
            if (dx == 0) == (dy == 0):
                continue
            out.append((x, y, direction(dx, dy)))
    return out


def make_question(rng, shown, asked):
    """Pick an unasked question about the statements shown so far, or None."""
    options = []
    for x, y, d in question_candidates(shown):
        options.append(f"Which direction is the {x} from the {y}?\t{d}")
        dd = rng.choice(DIRS)
        answer = "yes" if dd == d else "no"
        options.append(f"Is the {x} {dd} of the {y}?\t{answer}")
    options = [o for o in options if o.split("\t")[0] not in asked]
    if not options:
        return None
    text = rng.choice(options)
    asked.add(text.split("\t")[0])
    return text


def statement(rng, fact):
    a, d, b = fact
    if rng.random() < 0.5:
        return f"The {a} is {d} of the {b}."
    return f"The {b} lies {OPP[d]} of the {a}."


def try_story(rng):
    k = rng.randint(4, 7)
    cells = make_blob(rng, k)
    names = rng.sample(PLACES, k)
    name_of = dict(zip(cells, names))

    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            (x1, y1), (x2, y2) = cells[i], cells[j]
            if abs(x1 - x2) + abs(y1 - y2) == 1:
                edges.append((i, j))
    if len(edges) < 3:
        return None

    count = rng.randint(3, min(12, len(edges)))
    facts = []
    for i, j in rng.sample(edges, count):
        if rng.random() < 0.5:
            i, j = j, i
        # place i is in direction d from place j
        vx = cells[i][0] - cells[j][0]
        vy = cells[i][1] - cells[j][1]
        facts.append((name_of[cells[i]], VEC_TO_DIR[(vx, vy)], name_of[cells[j]]))
    rng.shuffle(facts)

    target = rng.randint(1, 5)
    lines, shown, asked = [], [], set()
    qcount = 0
    for fact in facts:
        lines.append(statement(rng, fact))
        shown.append(fact)
        if qcount < target and len(shown) >= 2 and rng.random() < 0.4:
            q = make_question(rng, shown, asked)
            if q is not None:
                lines.append(q)
                qcount += 1
    while qcount < target:
        q = make_question(rng, shown, asked)
        if q is None:
            break
        lines.append(q)
        qcount += 1
    if qcount == 0:
        return None
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for number, line in enumerate(make_story(rng), 1):
            out.append(f"{number} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
