import random
import sys

# Task: relative height ordering. Facts compare the heights of objects
# ("taller" / "shorter"). Facts are chained so that answers need transitive
# reasoning over several statements.

OBJECTS = ["vase", "lamp", "mug", "tower", "bottle", "candle", "stool", "jar"]


def closure_matrix(names, known):
    idx = {o: i for i, o in enumerate(names)}
    k = len(names)
    rel = [[False] * k for _ in range(k)]
    for a, b in known:
        rel[idx[a]][idx[b]] = True
    for m in range(k):
        for i in range(k):
            if rel[i][m]:
                for j in range(k):
                    if rel[m][j]:
                        rel[i][j] = True
    return rel, idx


def make_question(rng, known, names, pos, order):
    """Return (question_text, answer) if the answer follows from known facts."""
    rel, idx = closure_matrix(names, known)
    mentioned = {x for p in known for x in p}
    ml = [o for o in names if o in mentioned]  # fixed order, no set iteration
    k = len(names)
    if rng.random() < 0.7:
        if len(ml) < 2:
            return None
        a, b = rng.sample(ml, 2)
        if not (rel[idx[a]][idx[b]] or rel[idx[b]][idx[a]]):
            return None
        a_taller = pos[a] < pos[b]
        if rng.random() < 0.5:
            text = "Is the %s taller than the %s?" % (a, b)
            ans = "yes" if a_taller else "no"
        else:
            text = "Is the %s shorter than the %s?" % (a, b)
            ans = "yes" if not a_taller else "no"
        return text, ans
    # Superlative questions: only when every object of the story is mentioned
    # and the closure already shows a unique tallest / shortest.
    if len(ml) != k:
        return None
    if rng.random() < 0.5:
        t = order[0]
        if all(rel[idx[t]][idx[x]] for x in names if x != t):
            return "Which is the tallest?", t
    else:
        s = order[-1]
        if all(rel[idx[x]][idx[s]] for x in names if x != s):
            return "Which is the shortest?", s
    return None


def make_story(rng):
    k = rng.randint(3, 5)
    names = rng.sample(OBJECTS, k)
    order = names[:]
    rng.shuffle(order)  # order[0] is the tallest, order[-1] the shortest
    pos = {o: i for i, o in enumerate(order)}

    chain = [(order[i], order[i + 1]) for i in range(k - 1)]
    others = [(order[i], order[j]) for i in range(k) for j in range(i + 2, k)]
    n_stmt = rng.randint(max(3, k - 1), k * (k - 1) // 2)
    facts = chain + rng.sample(others, n_stmt - (k - 1))
    rng.shuffle(facts)

    body = []
    known = []
    questions = 0
    for idx_f, (a, b) in enumerate(facts):
        known.append((a, b))
        if rng.random() < 0.5:
            body.append("The %s is taller than the %s." % (a, b))
        else:
            body.append("The %s is shorter than the %s." % (b, a))
        if questions < 4 and idx_f < len(facts) - 1 and rng.random() < 0.35:
            q = make_question(rng, known, names, pos, order)
            if q is not None:
                body.append(q[0] + "\t" + q[1])
                questions += 1

    # Always end with at least one question, answerable from all statements.
    final = None
    for _ in range(200):
        final = make_question(rng, known, names, pos, order)
        if final is not None:
            break
    assert final is not None
    body.append(final[0] + "\t" + final[1])

    return ["%d %s" % (i, s) for i, s in enumerate(body, 1)]


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
