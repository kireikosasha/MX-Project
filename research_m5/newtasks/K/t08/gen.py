"""Synthetic places-on-a-map reading task.

Each story places a few named places on a grid. Statements say that one place
is directly north, south, east or west of another. Questions ask which place
lies in a direction from a given place, or which direction one place is from
another. Answers are computed from the simulated map and checked against what
the statements above the question determine.

Usage: python gen.py N SEED
"""
import random
import sys

PLACES = ["park", "school", "mill", "farm", "lake", "shop", "barn", "tower"]
DIRS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def make_map(rng):
    """Place 3 to 6 named places on distinct grid cells, each touching an earlier one."""
    names = rng.sample(PLACES, rng.randint(3, 6))
    cells = {names[0]: (0, 0)}
    taken = {(0, 0)}
    for name in names[1:]:
        while True:
            base = rng.choice(list(cells))
            vx, vy = DIRS[rng.choice(list(DIRS))]
            cell = (cells[base][0] + vx, cells[base][1] + vy)
            if cell not in taken:
                break
        cells[name] = cell
        taken.add(cell)
    return cells


def unit_edges(cells):
    """All pairs of places that touch on the map, as (a, d, b): a is directly d of b."""
    names = list(cells)
    edges = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dx = cells[a][0] - cells[b][0]
            dy = cells[a][1] - cells[b][1]
            for d, v in DIRS.items():
                if (dx, dy) == v:
                    edges.append((a, d, b))
                elif (-dx, -dy) == v:
                    edges.append((b, d, a))
    return edges


def components(facts):
    """Relative grid positions of places, one dict per connected group of statements."""
    adj = {}
    for a, d, b in facts:
        vx, vy = DIRS[d]
        adj.setdefault(a, []).append((b, -vx, -vy))  # b = a - d
        adj.setdefault(b, []).append((a, vx, vy))    # a = b + d
    comp = {}
    for start in adj:
        if start in comp:
            continue
        cmap = {start: (0, 0)}
        stack = [start]
        while stack:
            u = stack.pop()
            ux, uy = cmap[u]
            for v, dx, dy in adj[u]:
                if v not in cmap:
                    cmap[v] = (ux + dx, uy + dy)
                    stack.append(v)
        for n in cmap:
            comp[n] = cmap
    return comp


def candidates(facts):
    """Questions that the statements so far fully determine.

    Each candidate is (key, phrasings, answer). Questions need at least two
    statements of reasoning: the two places must not be stated directly.
    """
    comp = components(facts)
    linked = {frozenset((a, b)) for a, _, b in facts}
    result = []
    for x in comp:
        cmap = comp[x]
        px = cmap[x]
        inv = {p: n for n, p in cmap.items()}
        for d, (vx, vy) in DIRS.items():
            y = inv.get((px[0] + vx, px[1] + vy))
            if y is not None and frozenset((x, y)) not in linked:
                phr = ["What is directly %s of the %s?" % (d, x),
                       "Which place is directly %s of the %s?" % (d, x)]
                result.append((("A", x, d), phr, y))
        for y, py in cmap.items():
            if y == x:
                continue
            ox, oy = px[0] - py[0], px[1] - py[1]
            if (ox == 0 and abs(oy) >= 2) or (oy == 0 and abs(ox) >= 2):
                if ox == 0:
                    dname = "north" if oy > 0 else "south"
                else:
                    dname = "east" if ox > 0 else "west"
                phr = ["Which direction is the %s from the %s?" % (x, y)]
                result.append((("B", x, y), phr, dname))
    return result


def make_story(rng):
    while True:
        cells = make_map(rng)
        edges = unit_edges(cells)
        n_st = rng.randint(3, 12)
        order = []
        while len(order) < n_st:
            pool = edges[:]
            rng.shuffle(pool)
            order.extend(pool)
        facts = []
        for a, d, b in order[:n_st]:
            facts.append((a, d, b) if rng.random() < 0.5 else (b, OPP[d], a))

        n_q = rng.randint(1, 5)
        cuts = sorted(rng.randint(2, n_st) for _ in range(n_q))
        questions = {}
        seen = set()
        ok = True
        for t in cuts:
            options = [c for c in candidates(facts[:t]) if c[0] not in seen]
            if not options:
                ok = False
                break
            key, phr, ans = rng.choice(options)
            seen.add(key)
            text = rng.choice(phr)
            questions.setdefault(t, []).append((text, ans))
        if not ok:
            continue

        out = []
        for t in range(1, n_st + 1):
            a, d, b = facts[t - 1]
            out.append("The %s is directly %s of the %s." % (a, d, b))
            for text, ans in questions.get(t, []):
                out.append(text + "\t" + ans)
        return [str(i) + " " + line for i, line in enumerate(out, 1)]


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
