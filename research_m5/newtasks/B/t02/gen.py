import random
import sys

NAMES = ["ann", "bob", "cal", "dee"]
OBJECTS = ["ball", "book", "cup", "hat", "key"]


def gen_statement(rng, names, objs, owner, facts):
    free = [o for o in objs if owner[o] is None]
    held = [o for o in objs if owner[o] is not None]
    kinds = []
    if free:
        kinds.append("has")
    if held:
        kinds += ["gives", "drops", "swaps"]
    kind = rng.choice(kinds)

    if kind == "has":
        o = rng.choice(free)
        n = rng.choice(names)
        owner[o] = n
        if rng.random() < 0.5:
            text = f"{n} has the {o}."
        else:
            text = f"the {o} is with {n}."
        facts.append(({n}, {o}, False))
    elif kind == "gives":
        o = rng.choice(held)
        a = owner[o]
        b = rng.choice([x for x in names if x != a])
        owner[o] = b
        text = f"{a} gives the {o} to {b}."
        facts.append(({a, b}, {o}, False))
    elif kind == "drops":
        o = rng.choice(held)
        a = owner[o]
        owner[o] = None
        text = f"{a} drops the {o}."
        facts.append(({a}, {o}, False))
    else:
        holders = sorted({owner[o] for o in held})
        a = rng.choice(holders)
        b = rng.choice([x for x in names if x != a])
        for o in objs:
            if owner[o] == a:
                owner[o] = b
            elif owner[o] == b:
                owner[o] = a
        text = f"{a} swaps with {b}."
        facts.append(({a, b}, set(), True))
    return text


def gen_question(rng, names, objs, owner, facts):
    result = None
    for _ in range(8):
        kind = rng.choice(["who", "what", "does", "count"])
        qn, qo = set(), set()
        if kind == "who":
            o = rng.choice(objs)
            verb = rng.choice(["has", "holds"])
            text = f"who {verb} the {o}?"
            answer = owner[o] if owner[o] is not None else "nobody"
            qo = {o}
        else:
            n = rng.choice(names)
            qn = {n}
            held = sorted(o for o in objs if owner[o] == n)
            if kind == "what":
                text = f"what does {n} have?"
                answer = ",".join(held) if held else "nothing"
            elif kind == "does":
                o = rng.choice(objs)
                qo = {o}
                text = f"does {n} have the {o}?"
                answer = "yes" if owner[o] == n else "no"
            else:
                text = f"how many things does {n} have?"
                answer = str(len(held))
        rel = sum(1 for (fn, fo, sw) in facts
                  if (fn & qn) or (fo & qo) or (sw and qo))
        result = (text, answer)
        if rel >= 2:
            return result
    return result


def make_story(rng):
    names = rng.sample(NAMES, rng.choice([3, 4]))
    objs = rng.sample(OBJECTS, rng.choice([3, 4]))
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_stmt + n_q
    q_pos = set(rng.sample(range(1, total), n_q))  # first line is always a statement

    owner = {o: None for o in objs}
    facts = []
    lines = []
    for i in range(total):
        if i in q_pos:
            q, ans = gen_question(rng, names, objs, owner, facts)
            lines.append(f"{i + 1} {q}\t{ans}")
        else:
            s = gen_statement(rng, names, objs, owner, facts)
            lines.append(f"{i + 1} {s}")
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
