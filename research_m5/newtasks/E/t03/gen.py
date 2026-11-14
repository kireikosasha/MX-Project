"""Synthetic comparison-on-a-scale reasoning task generator.

Usage: python gen.py N SEED

Each story hides a total order of people by height. Statements reveal some
pairwise comparisons ("X is taller than Y" / "Y is shorter than X"). Questions
are asked only when the answer follows from the statements above them, using
transitive chains of comparisons. Answers are computed from the closure of the
statements seen so far.
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cara", "Dan", "Eli", "Fay", "Gus", "Hal"]


def stmt_text(rng, taller, shorter):
    if rng.random() < 0.5:
        return f"{taller} is taller than {shorter}."
    return f"{shorter} is shorter than {taller}."


def closure(ents, facts):
    """rel[(a, b)] is True when a is known to be taller than b."""
    rel = {(a, b): False for a in ents for b in ents}
    for a, b in facts:
        rel[(a, b)] = True
    for k in ents:
        for i in ents:
            if rel[(i, k)]:
                for j in ents:
                    if rel[(k, j)]:
                        rel[(i, j)] = True
    return rel


def candidate_questions(rng, facts):
    ents = sorted({x for f in facts for x in f})
    rel = closure(ents, facts)
    direct = set(facts)
    out = []

    # Yes/no comparisons that need a chain of statements.
    for a in ents:
        for b in ents:
            if a == b or (a, b) in direct or (b, a) in direct:
                continue
            if rel[(a, b)] or rel[(b, a)]:
                if rng.random() < 0.5:
                    out.append((f"Is {a} taller than {b}?",
                                "yes" if rel[(a, b)] else "no"))
                else:
                    out.append((f"Is {a} shorter than {b}?",
                                "yes" if rel[(b, a)] else "no"))

    # Superlatives and list answers, only when fully determined.
    for m in ents:
        others = [e for e in ents if e != m]
        if all(rel[(m, e)] for e in others) and \
                not all((m, e) in direct for e in others):
            out.append(("Who is the tallest?", m.lower()))
        if all(rel[(e, m)] for e in others) and \
                not all((e, m) in direct for e in others):
            out.append(("Who is the shortest?", m.lower()))
        if all(rel[(e, m)] or rel[(m, e)] for e in others):
            above = [e for e in others if rel[(e, m)]]
            if above and not all((e, m) in direct for e in above):
                out.append((f"Who is taller than {m}?",
                            ",".join(sorted(e.lower() for e in above))))
    return out


def ask_once(rng, facts, lines, used):
    """Append one new question that is answerable now. Return True if added."""
    cands = [c for c in candidate_questions(rng, facts) if c[0] not in used]
    if not cands:
        return False
    q = rng.choice(cands)
    lines.append(q)
    used.add(q[0])
    return True


def try_story(rng):
    cast = rng.sample(NAMES, rng.randint(3, 5))
    order = cast[:]
    rng.shuffle(order)
    rank = {name: i for i, name in enumerate(order)}  # higher = taller
    n_st = rng.randint(3, 12)
    q_target = rng.randint(1, 5)

    facts, lines, used = [], [], set()
    asked = 0
    while len(facts) < n_st:
        if facts and asked < q_target and rng.random() < 0.5:
            if ask_once(rng, facts, lines, used):
                asked += 1
                continue
        pairs = [(a, b) for a in cast for b in cast
                 if rank[a] > rank[b] and (a, b) not in facts]
        if not pairs:
            break
        a, b = rng.choice(pairs)
        facts.append((a, b))
        lines.append((stmt_text(rng, a, b), None))

    while facts and asked < q_target:
        if not ask_once(rng, facts, lines, used):
            break
        asked += 1

    if len(facts) < 3 or asked < 1:
        return None
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, (text, answer) in enumerate(make_story(rng), 1):
            if answer is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{answer}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
