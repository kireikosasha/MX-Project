import itertools
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eli", "fay"]
PLACES = ["hall", "park", "shop", "lab", "barn"]
MAX_ORDER_SET = 7


def closure(n, edges):
    """Transitive closure of the known 'a happened before b' facts."""
    less = [[False] * n for _ in range(n)]
    for a, b in edges:
        less[a][b] = True
    for k in range(n):
        for i in range(n):
            if less[i][k]:
                for j in range(n):
                    if less[k][j]:
                        less[i][j] = True
    return less


def valid_orders(ids, less):
    """All orders of the event ids that respect the known partial order."""
    if len(ids) > MAX_ORDER_SET:
        return None
    result = []
    for perm in itertools.permutations(ids):
        ok = all(not less[perm[j]][perm[i]]
                 for i in range(len(perm)) for j in range(i + 1, len(perm)))
        if ok:
            result.append(perm)
    return result


def first_values(ids, events, less, field):
    orders = valid_orders(ids, less)
    if orders is None:
        return None
    return {events[perm[0]][field] for perm in orders}


def make_question(rng, events, revealed, edges, less):
    kind = rng.randrange(4)

    if kind == 0:
        # where was X when Y went to the P?  -> place of X's latest event before Y's event
        x = rng.choice(NAMES)
        mine = [e for e in revealed if events[e]["p"] == x]
        others = [e for e in revealed if events[e]["p"] != x]
        if len(mine) < 2 or not others:
            return None
        t = rng.choice(others)
        tt = events[t]["t"]
        before = [e for e in mine if events[e]["t"] < tt]
        if not before:
            return None
        ans = events[max(before, key=lambda e: events[e]["t"])]["pl"]
        orders = valid_orders(mine + [t], less)
        if orders is None:
            return None
        seen = set()
        for perm in orders:
            pos = perm.index(t)
            seen.add(events[perm[pos - 1]]["pl"] if pos > 0 else None)
        if seen != {ans}:
            return None
        text = f"where was {x} when {events[t]['p']} went to the {events[t]['pl']}?"
        return text, ans

    if kind == 1:
        # did X go to the P before Y went to the Q?  -> yes / no
        if len(revealed) < 2:
            return None
        a, b = rng.sample(revealed, 2)
        if (a, b) in edges or (b, a) in edges:
            return None
        if not (less[a][b] or less[b][a]):
            return None
        ans = "yes" if events[a]["t"] < events[b]["t"] else "no"
        if (ans == "yes") != less[a][b]:
            return None
        text = (f"did {events[a]['p']} go to the {events[a]['pl']} before "
                f"{events[b]['p']} went to the {events[b]['pl']}?")
        return text, ans

    if kind == 2:
        # who went to the P first?  -> name
        pl = rng.choice(PLACES)
        ids = [e for e in revealed if events[e]["pl"] == pl]
        if len(ids) < 2:
            return None
        ans = events[min(ids, key=lambda e: events[e]["t"])]["p"]
        if first_values(ids, events, less, "p") != {ans}:
            return None
        return f"who went to the {pl} first?", ans

    # where did X go first?  -> place
    x = rng.choice(NAMES)
    ids = [e for e in revealed if events[e]["p"] == x]
    if len(ids) < 2:
        return None
    ans = events[min(ids, key=lambda e: events[e]["t"])]["pl"]
    if first_values(ids, events, less, "pl") != {ans}:
        return None
    return f"where did {x} go first?", ans


def try_story(rng):
    n = rng.randint(3, 12)
    q = rng.randint(1, 5)
    total = n + q
    pairs = [(p, pl) for p in NAMES for pl in PLACES]
    picked = rng.sample(pairs, n)
    rng.shuffle(picked)
    # hidden timeline: event i happened at time i + 1
    events = [{"p": p, "pl": pl, "t": i + 1} for i, (p, pl) in enumerate(picked)]
    reveal = list(range(n))
    rng.shuffle(reveal)  # statements are told out of order

    qpos = set(rng.sample(range(2, total), q))
    edges = []
    revealed = []
    at_events = []
    entries = []
    k = 0
    for pos in range(total):
        if pos in qpos:
            less = closure(n, edges)
            made = None
            for _ in range(20):
                made = make_question(rng, events, revealed, edges, less)
                if made:
                    break
            if not made:
                return None
            entries.append(made)
        else:
            e = reveal[k]
            k += 1
            ev = events[e]
            if not revealed or rng.random() < 0.4:
                text = f"{ev['p']} went to the {ev['pl']} at {ev['t']}."
                for f in at_events:
                    if ev["t"] < events[f]["t"]:
                        edges.append((e, f))
                    else:
                        edges.append((f, e))
                at_events.append(e)
            else:
                r = rng.choice(revealed)
                rv = events[r]
                if ev["t"] > rv["t"]:
                    edges.append((r, e))
                    word = "after"
                else:
                    edges.append((e, r))
                    word = "before"
                text = (f"{ev['p']} went to the {ev['pl']} {word} "
                        f"{rv['p']} went to the {rv['pl']}.")
            revealed.append(e)
            entries.append((text, None))
    return entries


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        entries = None
        while entries is None:
            entries = try_story(rng)
        for i, (text, ans) in enumerate(entries, 1):
            body = text[0].upper() + text[1:]
            lines.append(f"{i} {body}" + ("" if ans is None else "\t" + ans))
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
