#!/usr/bin/env python3
"""Generate synthetic height-comparison stories with yes/no and extreme questions.

Usage: python gen.py N SEED

Each story has a hidden set of people with distinct heights (the simulated
world). Statements reveal some pairwise comparisons. A question is asked only
when the statements above it determine its answer, and yes/no questions are
only asked about pairs that were not compared directly, so the answer needs
a transitive chain of at least two statements.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eva", "Finn",
         "Gus", "Hana", "Ivan", "Jada", "Kofi", "Lena"]
MIN_STATEMENTS, MAX_STATEMENTS = 3, 12
MIN_QUESTIONS, MAX_QUESTIONS = 1, 5
MAX_ATTEMPTS = 10000


def closure(facts):
    """Transitive closure of (taller, shorter) pairs."""
    rel = set(facts)
    while True:
        new = {(a, d) for (a, b) in rel for (c, d) in rel if b == c}
        if new <= rel:
            return rel
        rel |= new


def try_story(rng):
    n_stmt = rng.randint(MIN_STATEMENTS, MAX_STATEMENTS)
    n_q = rng.randint(MIN_QUESTIONS, MAX_QUESTIONS)
    n_people = rng.randint(3, 6)
    while n_people * (n_people - 1) // 2 < n_stmt:
        n_people += 1
    people = rng.sample(NAMES, n_people)
    # The simulated world: distinct heights for every person.
    heights = dict(zip(people, rng.sample(range(150, 200), n_people)))
    pairs = [(a, b) for i, a in enumerate(people) for b in people[i + 1:]]
    plan = ["S"] * n_stmt + ["Q"] * n_q
    rng.shuffle(plan)

    lines, facts, stated, asked = [], [], set(), set()
    for kind in plan:
        number = len(lines) + 1
        if kind == "S":
            avail = [p for p in pairs if frozenset(p) not in stated]
            if not avail:
                return None
            a, b = rng.choice(avail)
            if heights[a] < heights[b]:
                a, b = b, a  # a is the taller one
            stated.add(frozenset((a, b)))
            facts.append((a, b))
            if rng.random() < 0.5:
                text = f"{a} is taller than {b}."
            else:
                text = f"{b} is shorter than {a}."
            lines.append(f"{number} {text}")
        else:
            rel = closure(facts)
            options = []  # (question text, answer, answer agrees with simulation)
            for x in people:
                for y in people:
                    if x == y or frozenset((x, y)) in stated:
                        continue
                    if (x, y) in rel:
                        options.append((f"Is {x} taller than {y}?", "yes",
                                        heights[x] > heights[y]))
                    elif (y, x) in rel:
                        options.append((f"Is {x} taller than {y}?", "no",
                                        heights[x] < heights[y]))
            tallest = [p for p in people
                       if all((p, q) in rel for q in people if q != p)]
            shortest = [p for p in people
                        if all((q, p) in rel for q in people if q != p)]
            if tallest:
                t = tallest[0]
                options.append(("Who is the tallest?", t.lower(),
                                heights[t] == max(heights.values())))
            if shortest:
                s = shortest[0]
                options.append(("Who is the shortest?", s.lower(),
                                heights[s] == min(heights.values())))
            options = [o for o in options if o[0] not in asked]
            if not options:
                return None
            text, answer, ok = rng.choice(options)
            assert ok, "answer disagrees with the simulated world"
            asked.add(text)
            lines.append(f"{number} {text}\t{answer}")
    return lines


def make_story(rng):
    for _ in range(MAX_ATTEMPTS):
        lines = try_story(rng)
        if lines is not None:
            return lines
    raise RuntimeError("could not build a story")


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 2
    count, seed = int(argv[1]), int(argv[2])
    rng = random.Random(seed)
    for _ in range(count):
        for line in make_story(rng):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
