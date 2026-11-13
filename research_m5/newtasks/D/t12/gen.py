import random
import sys

# Task: household chore tracking. Statements assign chores to people or swap
# all chores between two people; questions ask who does a chore, whether a
# person does a chore, or which chores a person does. Answers come from
# simulating the chore assignment.

NAMES = ["Ann", "Ben", "Cal", "Dina", "Eli", "Fay", "Gus", "Hana"]
CHORES = ["floor", "bin", "plant", "car", "oven", "sink"]


def named(rec, p):
    if rec["kind"] == "assign":
        return rec["p"] == p
    return p in (rec["p"], rec["q"])


def relevant_for_person(stmts, p, c):
    # statements that mention the person, or assign the chore
    return sum(1 for s in stmts if named(s, p) or (s["kind"] == "assign" and s["c"] == c))


def try_question(rng, people, chores, owner, stmts):
    kind = rng.choice(["who", "does", "what"])
    if kind == "who":
        c = rng.choice(chores)
        n = sum(1 for s in stmts if s["kind"] == "assign" and s["c"] == c)
        if n < 2:
            return None
        ans = owner[c].lower() if owner[c] is not None else "nobody"
        return ("Who does the %s?" % c, ans)
    if kind == "does":
        p = rng.choice(people)
        c = rng.choice(chores)
        if relevant_for_person(stmts, p, c) < 2:
            return None
        ans = "yes" if owner[c] == p else "no"
        return ("Does %s do the %s?" % (p, c), ans)
    p = rng.choice(people)
    n = sum(1 for s in stmts if named(s, p))
    if n < 2:
        return None
    mine = sorted(c for c in chores if owner[c] == p)
    if not mine:
        return None
    return ("What does %s do?" % p, ",".join(mine))


def make_statement(rng, people, chores, owner, stmts):
    if rng.random() < 0.6:
        p = rng.choice(people)
        c = rng.choice(chores)
        owner[c] = p
        stmts.append({"kind": "assign", "p": p, "c": c})
        if rng.random() < 0.5:
            return "%s does the %s." % (p, c)
        return "The %s goes to %s." % (c, p)
    p, q = rng.sample(people, 2)
    for c in chores:
        if owner[c] == p:
            owner[c] = q
        elif owner[c] == q:
            owner[c] = p
    stmts.append({"kind": "swap", "p": p, "q": q})
    return "%s swaps chores with %s." % (p, q)


def build_story(rng):
    people = rng.sample(NAMES, rng.randint(3, 5))
    chores = rng.sample(CHORES, rng.randint(3, 5))
    owner = {c: None for c in chores}
    stmts = []
    target_s = rng.randint(3, 12)
    target_q = rng.randint(1, 5)
    items = []
    s_count = 0
    q_count = 0
    while s_count < target_s or q_count < target_q:
        if q_count < target_q and s_count >= 2 and (s_count == target_s or rng.random() < 0.4):
            q = try_question(rng, people, chores, owner, stmts)
            if q is not None:
                items.append(q)
                q_count += 1
                continue
            if s_count == target_s:
                return None
        if s_count < target_s:
            items.append((make_statement(rng, people, chores, owner, stmts), None))
            s_count += 1
    return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        items = None
        while items is None:
            items = build_story(rng)
        for i, (text, ans) in enumerate(items, 1):
            if ans is None:
                out.append("%d %s" % (i, text))
            else:
                out.append("%d %s\t%s" % (i, text, ans))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
