"""Synthetic reasoning task: event ordering over a day.

Each story lists some activities (each with a person) and their true order.
Statements give pairwise "before" facts. Questions ask about pairs whose order
follows only by chaining two or more statements (transitivity), and the answer
is computed from the transitive closure of the statements above the question.
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eve", "finn"]
ACTIVITIES = ["walk", "lunch", "bath", "nap", "reading", "cooking", "music", "chores"]


def transitive_closure(n, direct):
    before = [[False] * n for _ in range(n)]
    for a, b in direct:
        before[a][b] = True
    for k in range(n):
        for i in range(n):
            if before[i][k]:
                for j in range(n):
                    if before[k][j]:
                        before[i][j] = True
    return before


def make_story(rng):
    while True:
        k = rng.randint(3, 5)
        acts = rng.sample(ACTIVITIES, k)
        names = [rng.choice(NAMES) for _ in range(k)]
        phrases = ["%s of %s" % (acts[e], names[e]) for e in range(k)]

        # true order: order[pos] = event index at that position
        order = list(range(k))
        rng.shuffle(order)
        rank = [0] * k
        for pos, e in enumerate(order):
            rank[e] = pos

        n_facts = rng.randint(3, 8)
        n_questions = rng.randint(1, 5)
        rest = ["F"] * (n_facts - 1) + ["Q"] * n_questions
        rng.shuffle(rest)
        items = ["F"] + rest  # the first line is always a statement

        direct = set()
        lines = []
        for item in items:
            if item == "F":
                cands = [(a, b) for a in range(k) for b in range(k)
                         if rank[a] < rank[b] and (a, b) not in direct]
                if not cands:
                    continue
                a, b = rng.choice(cands)
                direct.add((a, b))
                pattern = rng.randrange(3)
                if pattern == 0:
                    text = "The %s is before the %s." % (phrases[a], phrases[b])
                elif pattern == 1:
                    text = "The %s comes before the %s." % (phrases[a], phrases[b])
                else:
                    text = "The %s is after the %s." % (phrases[b], phrases[a])
                lines.append(("S", text))
            else:
                closed = transitive_closure(k, direct)
                # pairs not stated directly, but ordered by chaining
                cands = [(a, b) for a in range(k) for b in range(a + 1, k)
                         if (a, b) not in direct and (b, a) not in direct
                         and (closed[a][b] or closed[b][a])]
                if not cands:
                    continue
                a, b = rng.choice(cands)
                if rng.random() < 0.5:
                    if rng.random() < 0.5:
                        a, b = b, a
                    text = "Is the %s before the %s?" % (phrases[a], phrases[b])
                    answer = "yes" if closed[a][b] else "no"
                else:
                    earlier = a if closed[a][b] else b
                    text = "Which of the %s and the %s is earlier?" % (phrases[a], phrases[b])
                    answer = acts[earlier]
                lines.append(("Q", text, answer))

        n_s = sum(1 for line in lines if line[0] == "S")
        n_q = len(lines) - n_s
        if 3 <= n_s <= 12 and 1 <= n_q <= 5:
            return lines


def main():
    if len(sys.argv) != 3:
        print("usage: python gen.py N SEED", file=sys.stderr)
        sys.exit(2)
    n_stories = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n_stories):
        lines = make_story(rng)
        for idx, line in enumerate(lines, 1):
            if line[0] == "S":
                out.append("%d %s" % (idx, line[1]))
            else:
                out.append("%d %s\t%s" % (idx, line[1], line[2]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
