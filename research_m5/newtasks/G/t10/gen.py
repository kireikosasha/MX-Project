import random
import sys
from collections import Counter

NAMES = ["Ann", "Bob", "Cara", "Dan", "Eli", "Fay", "Gus", "Hana"]
CLUBS = ["chess", "music", "garden", "sailing"]


def pick_action(rng, state, people, clubs):
    """Pick a random statement that is valid in the current state.

    Each person is in at most one club at a time, so a person with no club
    can only join, and a person in a club can only leave or move.
    """
    p = rng.choice(people)
    cur = state[p]
    if cur is None:
        return p, ("join", rng.choice(clubs))
    options = [("leave", cur)]
    options += [("move", cur, h) for h in clubs if h != cur]
    return p, rng.choice(options)


def apply(state, p, act):
    """Update the simulated world and return the statement text."""
    if act[0] == "join":
        state[p] = act[1]
        return f"{p} joins the {act[1]} club."
    if act[0] == "leave":
        state[p] = None
        return f"{p} leaves the {act[1]} club."
    state[p] = act[2]
    return f"{p} moves from the {act[1]} club to the {act[2]} club."


def make_question(rng, state, people, mentioned, counts, clubs):
    """Return (question_text, answer) computed from the current state."""
    kind = rng.randrange(3)
    if kind == 0:
        # Prefer a person who has been the subject of at least two statements,
        # so the answer needs more than one sentence.
        cands = [p for p in mentioned if counts[p] >= 2] or sorted(mentioned)
        p = rng.choice(sorted(cands))
        g = state[p]
        return f"Which club is {p} in?", (g if g is not None else "none")
    g = rng.choice(clubs)
    members = sorted(p for p in people if state[p] == g)
    if kind == 1:
        answer = ",".join(m.lower() for m in members) or "none"
        return f"Who is in the {g} club?", answer
    return f"How many are in the {g} club?", str(len(members))


def make_story(rng):
    people = rng.sample(NAMES, rng.randint(3, 5))
    clubs = rng.sample(CLUBS, rng.randint(2, 4))
    state = {p: None for p in people}
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, min(5, n_st))
    slots = set(rng.sample(range(1, n_st + 1), n_q))

    mentioned = set()
    counts = Counter()
    items = []
    for k in range(1, n_st + 1):
        p, act = pick_action(rng, state, people, clubs)
        mentioned.add(p)
        counts[p] += 1
        items.append(apply(state, p, act))
        if k in slots:
            items.append(make_question(rng, state, people, mentioned, counts, clubs))
    return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, item in enumerate(make_story(rng), 1):
            if isinstance(item, tuple):
                out.append(f"{i} {item[0]}\t{item[1]}")
            else:
                out.append(f"{i} {item}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
