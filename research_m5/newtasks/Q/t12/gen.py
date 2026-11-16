import random
import sys

# Chore nouns. Their index is their hidden rank in a story's schedule.
NOUNS = ["rug", "pot", "bed", "sink", "bin", "wall", "floor", "fence"]


def closure(k, pairs):
    """Transitive closure: less[i][j] is True when i is known to come before j."""
    less = [[False] * k for _ in range(k)]
    for a, b in pairs:
        less[a][b] = True
    for m in range(k):
        for i in range(k):
            if less[i][m]:
                for j in range(k):
                    if less[m][j]:
                        less[i][j] = True
    return less


def question_options(chores, direct, less, mentioned):
    """Questions whose answer is fixed by the statements so far, as (text, answer)."""
    items = sorted(mentioned)
    opts = []
    # Yes/no ordering questions: the order must follow from a chain, not one sentence.
    for a in items:
        for b in items:
            if a == b or (a, b) in direct or (b, a) in direct:
                continue
            if less[a][b]:
                opts.append(("Does the %s come before the %s?" % (chores[a], chores[b]), "yes"))
            elif less[b][a]:
                opts.append(("Does the %s come before the %s?" % (chores[a], chores[b]), "no"))
    # First / last questions: one chore must be known to lie before (after) all others.
    if len(items) >= 3:
        for x in items:
            if all(y == x or less[x][y] for y in items):
                if sum((x, y) in direct for y in items) < len(items) - 1:
                    opts.append(("Which one comes first?", chores[x]))
            if all(y == x or less[y][x] for y in items):
                if sum((y, x) in direct for y in items) < len(items) - 1:
                    opts.append(("Which one comes last?", chores[x]))
    return opts


def try_story(rng):
    k = rng.randint(4, 6)
    chores = rng.sample(NOUNS, k)  # index order is the hidden schedule
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    direct = set()
    lines = []
    asked = set()
    while n_stmt or n_q:
        less = closure(k, direct)
        mentioned = {i for pair in direct for i in pair}
        if n_q and (n_stmt == 0 or (direct and rng.random() < 0.45)):
            opts = [o for o in question_options(chores, direct, less, mentioned)
                    if o[0] not in asked]
            if opts:
                text, ans = rng.choice(opts)
                asked.add(text)
                lines.append("%s\t%s" % (text, ans))
                n_q -= 1
                continue
            if n_stmt == 0:
                return None
        # Prefer a pair whose order is not yet known.
        cand = [(i, j) for i in range(k) for j in range(i + 1, k)
                if not less[i][j] and not less[j][i]]
        if not cand:
            cand = [(i, j) for i in range(k) for j in range(i + 1, k)]
        i, j = rng.choice(cand)
        if rng.random() < 0.5:
            text = "The %s comes before the %s." % (chores[i], chores[j])
        else:
            text = "The %s comes after the %s." % (chores[j], chores[i])
        direct.add((i, j))
        lines.append(text)
        n_stmt -= 1
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
        for number, line in enumerate(make_story(rng), 1):
            out.append("%d %s" % (number, line))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
