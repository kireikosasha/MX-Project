import itertools
import random
import sys

# Scale task: each story is a set of pairwise strength comparisons among a
# small roster of names. Questions need several comparisons chained together.

NAMES = ["kesh", "orla", "brynn", "tavi", "moxa", "lenn", "pru", "zade", "dorn", "ilse"]

STATEMENT_PATTERNS = (
    "{a} exceeds {b} in strength.",
    "{b} trails {a} in strength.",
    "{a} outdoes {b} in strength.",
)


def cap(word):
    return word[:1].upper() + word[1:]


def answer(q, rank, roster):
    """Answer of question q under one candidate strength ordering (rank: name -> int)."""
    kind = q[0]
    if kind == "exceed":
        return "yes" if rank[q[1]] > rank[q[2]] else "no"
    if kind == "trail":
        return "yes" if rank[q[1]] < rank[q[2]] else "no"
    if kind == "top":
        return max(roster, key=lambda n: rank[n])
    if kind == "bottom":
        return min(roster, key=lambda n: rank[n])
    if kind == "count":
        return str(sum(1 for n in roster if rank[n] > rank[q[1]]))
    # "order"
    return ",".join(sorted(roster, key=lambda n: rank[n]))


def question_text(q):
    kind = q[0]
    if kind == "exceed":
        return "Does %s exceed %s?" % (cap(q[1]), cap(q[2]))
    if kind == "trail":
        return "Does %s trail %s?" % (cap(q[1]), cap(q[2]))
    if kind == "top":
        return "Which name stands strongest?"
    if kind == "bottom":
        return "Which name stands weakest?"
    if kind == "count":
        return "How many names exceed %s?" % cap(q[1])
    return "Order the names from weakest to strongest?"


def random_question(rng, roster):
    a, b = rng.sample(roster, 2)
    return rng.choice([
        ("exceed", a, b),
        ("trail", a, b),
        ("top",),
        ("bottom",),
        ("count", a),
        ("order",),
    ])


def pick_question(rng, roster, alive, stmt_sets, truth, tries=30):
    """Return (question, answer) that is fixed by all statements so far and
    not fixed by any single statement alone, or None."""
    for _ in range(tries):
        q = random_question(rng, roster)
        want = answer(q, truth, roster)
        # Answer must be the same for every ordering still consistent with the statements.
        if any(answer(q, p, roster) != want for p in alive):
            continue
        # Reject questions a single statement already settles.
        if any(all(answer(q, p, roster) == want for p in s) for s in stmt_sets):
            continue
        return q, want
    return None


def build_story(rng):
    k = rng.randint(4, 5)
    roster = rng.sample(NAMES, k)
    order = roster[:]
    rng.shuffle(order)
    truth = {n: i for i, n in enumerate(order)}
    all_perms = [dict(zip(p, range(k))) for p in itertools.permutations(roster)]

    target_s = rng.randint(3, 12)
    target_q = rng.randint(1, 5)

    lines = []        # (text, answer or None for statements)
    stmt_sets = []    # for each statement, the orderings it allows
    alive = all_perms # orderings consistent with all statements so far
    n_s = 0
    n_q = 0

    while n_s < target_s:
        if n_s >= 1 and n_q < target_q and rng.random() < 0.35:
            found = pick_question(rng, roster, alive, stmt_sets, truth)
            if found:
                q, want = found
                lines.append((question_text(q), want))
                n_q += 1
        x, y = rng.sample(roster, 2)
        a, b = (x, y) if truth[x] > truth[y] else (y, x)
        text = rng.choice(STATEMENT_PATTERNS).format(a=cap(a), b=cap(b))
        stmt_sets.append([p for p in all_perms if p[a] > p[b]])
        alive = [p for p in alive if p[a] > p[b]]
        lines.append((text, None))
        n_s += 1

    while n_q < target_q:
        found = pick_question(rng, roster, alive, stmt_sets, truth)
        if not found:
            break
        q, want = found
        lines.append((question_text(q), want))
        n_q += 1

    if n_q == 0:
        return None
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = None
        while story is None:
            story = build_story(rng)
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                out.append("%d %s" % (i, text))
            else:
                out.append("%d %s\t%s" % (i, text, ans))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
