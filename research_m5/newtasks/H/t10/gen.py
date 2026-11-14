import random
import sys

# Task: group membership tracking. People join, leave and move between
# groups; questions ask which group a person is in, whether a person is in a
# given group, who is in a group, or how many are in it. Answers come from
# simulating the statements above each question.

NAMES = ["alice", "bob", "carol", "dan", "eve", "frank", "grace", "henry"]
GROUPS = ["red", "blue", "green", "gold"]


def cap(word):
    return word[0].upper() + word[1:]


def make_story(rng):
    num_st = rng.randint(3, 12)
    num_q = rng.randint(1, 5)
    people = rng.sample(NAMES, rng.randint(3, 5))
    groups = rng.sample(GROUPS, rng.randint(2, 4))

    # state maps person -> group name or None (not in any group)
    state = {p: None for p in people}
    snaps = [dict(state)]
    mentioned = [set()]
    seen = set()
    stmts = []
    for _ in range(num_st):
        options = []
        for p in people:
            cur = state[p]
            if cur is None:
                for g in groups:
                    options.append(("join", p, g))
            else:
                options.append(("leave", p, cur))
                for g in groups:
                    if g != cur:
                        options.append(("move", p, g))
        kind, p, g = rng.choice(options)
        if kind == "join":
            state[p] = g
            text = f"{cap(p)} joins the {g} group."
        elif kind == "leave":
            state[p] = None
            text = f"{cap(p)} leaves the {g} group."
        else:
            state[p] = g
            text = f"{cap(p)} moves to the {g} group."
        stmts.append(text)
        seen.add(p)
        snaps.append(dict(state))
        mentioned.append(set(seen))

    # questions are placed after at least 2 statements, answered from the
    # state at that point
    slots = {}
    used = set()
    made = 0
    attempts = 0
    while made < num_q and attempts < 200:
        attempts += 1
        k = rng.randint(2, num_st)
        p = rng.choice(sorted(mentioned[k]))
        g = rng.choice(groups)
        snap = snaps[k]
        t = rng.randrange(4)
        if t == 0:
            text = f"Which group is {cap(p)} in?"
            ans = snap[p] or "none"
        elif t == 1:
            text = f"Is {cap(p)} in the {g} group?"
            ans = "yes" if snap[p] == g else "no"
        elif t == 2:
            text = f"Who is in the {g} group?"
            members = sorted(x for x in people if snap[x] == g)
            ans = ",".join(members) if members else "none"
        else:
            text = f"How many are in the {g} group?"
            ans = str(sum(1 for x in people if snap[x] == g))
        if text in used:
            continue
        used.add(text)
        slots.setdefault(k, []).append((text, ans))
        made += 1

    lines = []
    for k in range(num_st + 1):
        if k > 0:
            lines.append(stmts[k - 1])
        for text, ans in slots.get(k, []):
            lines.append(text + "\t" + ans)
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
