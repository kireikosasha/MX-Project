import random
import sys

# Quantities task: people hold numbers of things, gain, lose, give and take
# some, and questions ask how many someone has or compare amounts.

NAMES = ["Ann", "Bob", "Cara", "Dan", "Eve"]
THINGS = ["apples", "coins", "pens", "balls"]


def apply_op(state, idx, op):
    counts, deps = state
    kind, p, q, t, amt = op
    if kind == "has":
        counts[(p, t)] = amt
        deps[(p, t)] = [idx]
    elif kind == "gets":
        counts[(p, t)] += amt
        deps[(p, t)].append(idx)
    elif kind == "loses":
        counts[(p, t)] -= amt
        deps[(p, t)].append(idx)
    elif kind == "gives":
        counts[(p, t)] -= amt
        counts[(q, t)] += amt
        deps[(p, t)].append(idx)
        deps[(q, t)].append(idx)
    elif kind == "takes":
        counts[(q, t)] -= amt
        counts[(p, t)] += amt
        deps[(p, t)].append(idx)
        deps[(q, t)].append(idx)


def op_text(op):
    kind, p, q, t, amt = op
    if kind == "has":
        return f"{p} has {amt} {t}."
    if kind == "gets":
        return f"{p} gets {amt} {t}."
    if kind == "loses":
        return f"{p} loses {amt} {t}."
    if kind == "gives":
        return f"{p} gives {amt} {t} to {q}."
    return f"{p} takes {amt} {t} from {q}."


def union_deps(deps, keys):
    s = set()
    for k in keys:
        s.update(deps[k])
    return s


def make_question(rng, state, people, things):
    counts, deps = state
    for _ in range(30):
        kind = rng.choice(["how_many", "most", "more", "any", "all", "list"])
        if kind == "how_many":
            p = rng.choice(people)
            t = rng.choice(things)
            if len(set(deps[(p, t)])) < 2:
                continue
            return (f"How many {t} does {p} have?", str(counts[(p, t)]))
        if kind == "most":
            t = rng.choice(things)
            vals = {p: counts[(p, t)] for p in people}
            mx = max(vals.values())
            winners = [p for p in people if vals[p] == mx]
            if len(winners) != 1:
                continue
            ds = union_deps(deps, [(p, t) for p in people])
            if len(ds) < 2:
                continue
            return (f"Who has the most {t}?", winners[0].lower())
        if kind == "more":
            p, q = rng.sample(people, 2)
            t = rng.choice(things)
            ds = union_deps(deps, [(p, t), (q, t)])
            if len(ds) < 2:
                continue
            ans = "yes" if counts[(p, t)] > counts[(q, t)] else "no"
            return (f"Does {p} have more {t} than {q}?", ans)
        if kind == "any":
            p = rng.choice(people)
            t = rng.choice(things)
            if len(set(deps[(p, t)])) < 2:
                continue
            ans = "yes" if counts[(p, t)] > 0 else "no"
            return (f"Does {p} have any {t}?", ans)
        if kind == "all":
            t = rng.choice(things)
            ds = union_deps(deps, [(p, t) for p in people])
            if len(ds) < 2:
                continue
            total = sum(counts[(p, t)] for p in people)
            return (f"How many {t} are there in all?", str(total))
        if kind == "list":
            if len(things) != 2:
                continue
            p = rng.choice(people)
            ds = union_deps(deps, [(p, t) for t in things])
            if len(ds) < 2:
                continue
            lst = sorted(t for t in things if counts[(p, t)] > 0)
            if len(lst) < 2:
                continue
            return (f"What things does {p} have?", ",".join(lst))
    return None


def try_story(rng):
    people = rng.sample(NAMES, rng.choice([2, 3]))
    things = rng.sample(THINGS, rng.choice([1, 2]))
    pairs = [(p, t) for p in people for t in things]
    rng.shuffle(pairs)

    state = ({}, {})
    counts = state[0]
    ops = []
    for p, t in pairs:
        op = ("has", p, None, t, rng.randint(2, 9))
        apply_op(state, len(ops), op)
        ops.append(op)
    n_has = len(ops)

    lo = max(1, 3 - n_has)
    hi = min(8, 12 - n_has)
    n_ev = rng.randint(lo, hi)
    for _ in range(n_ev):
        op = None
        for _attempt in range(50):
            kind = rng.choice(["gets", "loses", "gives", "takes"])
            p = rng.choice(people)
            q = rng.choice([x for x in people if x != p])
            t = rng.choice(things)
            amt = rng.randint(2, 4)
            if kind == "gets":
                cand = ("gets", p, None, t, amt)
            elif kind == "loses":
                if counts[(p, t)] < amt:
                    continue
                cand = ("loses", p, None, t, amt)
            elif kind == "gives":
                if counts[(p, t)] < amt:
                    continue
                cand = ("gives", p, q, t, amt)
            else:
                if counts[(q, t)] < amt:
                    continue
                cand = ("takes", p, q, t, amt)
            op = cand
            break
        if op is None:
            return None
        apply_op(state, len(ops), op)
        ops.append(op)

    # Questions go after all the initial facts, at random points.
    n_q = rng.randint(1, 5)
    positions = sorted(rng.randint(n_has, len(ops)) for _ in range(n_q))
    replay = ({}, {})
    cur = 0
    qs = []
    for pos in positions:
        while cur < pos:
            apply_op(replay, cur, ops[cur])
            cur += 1
        qa = make_question(rng, replay, people, things)
        if qa is not None:
            qs.append((pos, qa[0], qa[1]))
    if not qs:
        return None

    lines = []
    qi = 0
    for i in range(len(ops) + 1):
        while qi < len(qs) and qs[qi][0] == i:
            lines.append(f"{qs[qi][1]}\t{qs[qi][2]}")
            qi += 1
        if i < len(ops):
            lines.append(op_text(ops[i]))
    return lines


def make_story(rng):
    for _ in range(10000):
        lines = try_story(rng)
        if lines:
            return lines
    raise RuntimeError("could not generate a story")


def main():
    if len(sys.argv) != 3:
        print("usage: python gen.py N SEED", file=sys.stderr)
        sys.exit(1)
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lines = make_story(rng)
        for i, line in enumerate(lines, 1):
            out.append(f"{i} {line}")
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    main()
