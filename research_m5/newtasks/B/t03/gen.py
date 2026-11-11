"""Synthetic height-comparison reading task (transitive reasoning on one scale).

Usage: python gen.py N SEED
Prints N stories. Each story states pairwise "taller than" facts about a few
people, then asks yes/no, superlative and list questions answered only from the
statements above them. Answers come from a transitive closure of the facts.
"""

import random
import sys

NAMES = ["ann", "bob", "cal", "dan", "eve", "fay", "gus", "hal", "ivy", "jon"]


def closure(people, facts):
    """Transitive closure of 'x is taller than y' facts over the given people."""
    rel = set(facts)
    for k in people:
        for i in people:
            if (i, k) in rel:
                for j in people:
                    if (k, j) in rel:
                        rel.add((i, j))
    return rel


def relation(rel, a, b):
    """1 if a is known to be taller than b, -1 if known shorter, 0 if unknown."""
    if (a, b) in rel:
        return 1
    if (b, a) in rel:
        return -1
    return 0


def candidates(ment, facts, direct):
    """All questions that the statements so far determine unambiguously."""
    rel = closure(ment, facts)
    cands = []

    # yes/no about a pair never compared directly, but decided by a chain
    for a in ment:
        for b in ment:
            if a == b or (a, b) in direct or (b, a) in direct:
                continue
            r = relation(rel, a, b)
            if r:
                cands.append((f"is {a} taller than {b}?", "yes" if r == 1 else "no"))

    # superlatives: only when exactly one person is not beaten by anyone (or beats no one)
    tops = [a for a in ment if not any((b, a) in rel for b in ment)]
    if len(tops) == 1:
        cands.append(("who is the tallest?", tops[0]))
    bots = [a for a in ment if not any((a, b) in rel for b in ment)]
    if len(bots) == 1:
        cands.append(("who is the shortest?", bots[0]))

    # lists: only when every other person's height relation to x is known
    for x in ment:
        others = [y for y in ment if y != x]
        if any(relation(rel, x, y) == 0 for y in others):
            continue
        taller = [y for y in others if relation(rel, y, x) == 1]
        shorter = [y for y in others if relation(rel, x, y) == 1]
        if len(taller) >= 2:
            cands.append((f"who is taller than {x}?", ",".join(sorted(taller))))
        if len(shorter) >= 2:
            cands.append((f"who is shorter than {x}?", ",".join(sorted(shorter))))
    return cands


def try_story(rng):
    k = rng.randint(3, 5)
    people = rng.sample(NAMES, k)
    rng.shuffle(people)  # people[0] is the hidden tallest, people[-1] the shortest
    pos = {p: i for i, p in enumerate(people)}
    n_pairs = k * (k - 1) // 2
    n_stmt = rng.randint(3, min(8, n_pairs))  # distinct pairs, so no repeated fact
    n_q = rng.randint(1, 4)
    lines = []
    direct = set()  # (taller, shorter) pairs stated directly
    facts = set()
    asked = set()
    s_done = 0
    q_done = 0
    while s_done < n_stmt or q_done < n_q:
        if q_done < n_q and s_done >= 2 and (s_done == n_stmt or rng.random() < 0.4):
            ment = sorted({p for f in facts for p in f})
            cands = [c for c in candidates(ment, facts, direct) if c[0] not in asked]
            if cands:
                choice = rng.choice(cands)
                asked.add(choice[0])
                lines.append(choice)
                q_done += 1
                continue
            if s_done == n_stmt:
                return None
        if s_done >= n_stmt:
            return None
        pairs = [(x, y) for x in people for y in people if pos[x] < pos[y]]
        fresh = [p for p in pairs if p not in direct]
        x, y = rng.choice(fresh or pairs)
        if rng.random() < 0.5:
            text = f"{x} is taller than {y}."
        else:
            text = f"{y} is shorter than {x}."
        direct.add((x, y))
        facts.add((x, y))
        lines.append((text, None))
        s_done += 1
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        idx = 0
        for text, ans in make_story(rng):
            idx += 1
            line = f"{idx} {text}"
            if ans is not None:
                line += "\t" + ans
            out.append(line)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
