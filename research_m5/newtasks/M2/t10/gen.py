"""Synthetic group-membership reading-comprehension generator.

People join, leave and move between groups (teams, clubs, crews). Questions ask
whether a person is in a group, who is in a group, how many are in it, or which
group a person is in. Answers come from simulating the state of the world.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "ben", "cal", "dan", "eve", "fay", "gus", "hal"]
GROUPS = ["north", "south", "east", "west"]


def cap(word):
    return word[:1].upper() + word[1:]


def statement(rng, persons, member):
    free = [p for p in persons if member[p] is None]
    inside = [p for p in persons if member[p] is not None]
    actions = []
    if free:
        actions.append("join")
    if inside:
        actions.append("leave")
        actions.append("move")
    act = rng.choice(actions)

    if act == "join":
        p = rng.choice(free)
        g = rng.choice(GROUPS)
        member[p] = g
        return f"{cap(p)} joined the {g} group."
    if act == "leave":
        p = rng.choice(inside)
        g = member[p]
        member[p] = None
        return f"{cap(p)} left the {g} group."
    p = rng.choice(inside)
    g1 = member[p]
    g2 = rng.choice([g for g in GROUPS if g != g1])
    member[p] = g2
    return f"{cap(p)} moved from the {g1} group to the {g2} group."


def question(rng, persons, member):
    inside = [p for p in persons if member[p] is not None]
    occupied = sorted({member[p] for p in inside})
    kinds = ["is", "how"]
    if inside:
        kinds.append("which")
    if occupied:
        kinds.append("who")
    kind = rng.choice(kinds)

    if kind == "is":
        p = rng.choice(persons)
        g = rng.choice(GROUPS)
        answer = "yes" if member[p] == g else "no"
        return f"Is {cap(p)} in the {g} group?", answer
    if kind == "how":
        g = rng.choice(GROUPS)
        count = sum(1 for p in persons if member[p] == g)
        return f"How many people are in the {g} group?", str(count)
    if kind == "which":
        p = rng.choice(inside)
        return f"Which group is {cap(p)} in?", member[p]
    g = rng.choice(occupied)
    members = sorted(p for p in persons if member[p] == g)
    return f"Who is in the {g} group?", ",".join(members)


def story(rng):
    persons = rng.sample(NAMES, rng.randint(4, 6))
    member = {p: None for p in persons}
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_stmt + n_q
    # Question lines never come first, so every question has statements above it.
    q_slots = set(rng.sample(range(1, total), n_q))

    lines = []
    for i in range(total):
        if i in q_slots:
            text, answer = question(rng, persons, member)
            lines.append(f"{i + 1} {text}\t{answer}")
        else:
            text = statement(rng, persons, member)
            lines.append(f"{i + 1} {text}")
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
