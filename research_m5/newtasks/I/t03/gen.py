#!/usr/bin/env python3
"""Synthetic reading-comprehension task: comparison on one scale (taller / shorter).

Usage: python gen.py N SEED

Each story has a hidden height order of a few people. Statements tell pairwise
height facts that are true in that order. Questions are answered from the
transitive closure of the statements above them, and are only asked when the
answer is determined and needs at least two statements to derive.
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eva", "finn",
         "gina", "hugo", "ivy", "jack", "kim", "leo"]


def cap(word):
    return word.capitalize()


def closure(direct):
    """Transitive closure of 'taller than'; a pair (a, b) means a is taller than b."""
    rel = set(direct)
    changed = True
    while changed:
        changed = False
        for a, b in list(rel):
            for c, d in list(rel):
                if b == c and (a, d) not in rel:
                    rel.add((a, d))
                    changed = True
    return rel


def find_question(rng, mentioned, direct, rel, pos):
    """Return (question text, answer) determined by the statements so far, or None.

    pos[name] is the hidden height rank (0 = tallest); used only as a check.
    """
    names = sorted(mentioned)
    if len(names) < 2:
        return None
    for _ in range(80):
        kind = rng.choice(["yesno", "tallest", "shortest", "taller_than", "shorter_than"])

        if kind == "yesno":
            x, y = rng.sample(names, 2)
            if (x, y) in rel and (x, y) not in direct:
                ans = "yes"
            elif (y, x) in rel and (y, x) not in direct:
                ans = "no"
            else:
                continue
            assert ans == ("yes" if pos[x] < pos[y] else "no")
            return f"Is {cap(x)} taller than {cap(y)}?", ans

        if kind in ("tallest", "shortest"):
            if kind == "tallest":
                tops = [t for t in names if all((t, y) in rel for y in names if y != t)]
            else:
                tops = [t for t in names if all((y, t) in rel for y in names if y != t)]
            if len(tops) != 1:
                continue
            t = tops[0]
            if kind == "tallest":
                derived = any((t, y) not in direct for y in names if y != t)
                truth = min(names, key=lambda n: pos[n])
                text = "Who is the tallest?"
            else:
                derived = any((y, t) not in direct for y in names if y != t)
                truth = max(names, key=lambda n: pos[n])
                text = "Who is the shortest?"
            if not derived:
                continue
            assert truth == t
            return text, t

        # List questions: every other mentioned person must be comparable with x.
        x = rng.choice(names)
        others = [y for y in names if y != x]
        if not all((x, y) in rel or (y, x) in rel for y in others):
            continue
        if kind == "taller_than":
            ans = [y for y in others if (y, x) in rel]
            derived = any((y, x) not in direct for y in ans)
            text = f"Who is taller than {cap(x)}?"
            truth = sorted(y for y in others if pos[y] < pos[x])
        else:
            ans = [y for y in others if (x, y) in rel]
            derived = any((x, y) not in direct for y in ans)
            text = f"Who is shorter than {cap(x)}?"
            truth = sorted(y for y in others if pos[y] > pos[x])
        if not ans or not derived:
            continue
        assert truth == sorted(ans)
        return text, ",".join(sorted(ans))
    return None


def try_story(rng):
    k = rng.randint(3, 5)
    people = rng.sample(NAMES, k)
    order = people[:]
    rng.shuffle(order)
    pos = {p: i for i, p in enumerate(order)}  # hidden heights: smaller rank = taller

    n_statements = rng.randint(3, 12)
    n_questions = rng.randint(1, 5)
    events = ["S"] * (n_statements - 1) + ["Q"] * n_questions
    rng.shuffle(events)
    events = ["S"] + events  # the first sentence is always a statement

    direct = set()
    mentioned = set()
    lines = []
    for ev in events:
        number = len(lines) + 1
        rel = closure(direct)
        if ev == "S":
            pairs = [(a, b) for a in people for b in people if pos[a] < pos[b]]
            fresh = [p for p in pairs if p not in rel]
            a, b = rng.choice(fresh or pairs)
            direct.add((a, b))
            mentioned.update((a, b))
            if rng.random() < 0.5:
                text = f"{cap(a)} is taller than {cap(b)}."
            else:
                text = f"{cap(b)} is shorter than {cap(a)}."
            lines.append(f"{number} {text}")
        else:
            q = find_question(rng, mentioned, direct, rel, pos)
            if q is None:
                return None
            text, ans = q
            lines.append(f"{number} {text}\t{ans}")
    return lines


def make_story(rng):
    while True:
        story = try_story(rng)
        if story is not None:
            return story


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 1
    n = int(argv[1])
    seed = int(argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
