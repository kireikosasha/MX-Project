"""Synthetic chain-of-relation reading task generator.

Each story is a small forest of people linked by one relation per story
(either "parent" or "boss"; every person has at most one parent/boss).
Statements give single links (child -> parent) and root facts ("has no
parent"). Questions ask about two-step chains (grandparent, boss of the
boss), about the top of a chain, or yes/no versions of two-step chains.
Answers are computed by walking only the links stated above each question.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eli", "fay",
         "gus", "hana", "ivan", "jo", "kim", "leo"]


def new_world(rng):
    rel = rng.choice(["parent", "boss"])
    n = rng.randint(3, 10)
    names = rng.sample(NAMES, n)
    r = rng.randint(1, 2)
    parent = {}
    for i in range(r, n):
        parent[names[i]] = names[rng.randrange(i)]
    roots = names[:r]
    return rel, names, parent, roots


def capitalize(text):
    return text[0].upper() + text[1:]


def statement_text(rel, fact, rng):
    if fact[0] == "root":
        return capitalize("%s has no %s." % (fact[1], rel))
    child, par = fact[1], fact[2]
    form = rng.randrange(2)
    if rel == "parent":
        if form == 0:
            return capitalize("%s is the parent of %s." % (par, child))
        return capitalize("%s is the child of %s." % (child, par))
    if form == 0:
        return capitalize("%s is the boss of %s." % (par, child))
    return capitalize("%s reports to %s." % (child, par))


def question_text(rel, q):
    if q[0] == "top":
        return "Who is at the top above %s?" % q[1]
    if q[0] == "two":
        if rel == "parent":
            return "Who is the grandparent of %s?" % q[1]
        return "Who is the boss of the boss of %s?" % q[1]
    if rel == "parent":
        return "Is %s the grandparent of %s?" % (q[1], q[2])
    return "Is %s the boss of the boss of %s?" % (q[1], q[2])


def answer(q, kp, kr):
    """Answer q from the known links kp (child -> parent) and known roots kr.

    Returns (answer, number_of_statements_used) or None if not determined.
    """
    if q[0] == "top":
        cur, steps = q[1], 0
        while True:
            if cur in kr:
                return cur, steps + 1
            if cur not in kp:
                return None
            cur = kp[cur]
            steps += 1
    if q[0] == "two":
        p = kp.get(q[1])
        if p is None:
            return None
        g = kp.get(p)
        if g is None:
            return None
        return g, 2
    a, b = q[1], q[2]
    p = kp.get(b)
    if p is None:
        return None
    g = kp.get(p)
    if g is None:
        return None
    return ("yes" if g == a else "no"), 2


def random_question(rng, names, kp, kr, rel, used):
    for _ in range(30):
        kind = rng.choice(["two", "top", "yn"])
        x = rng.choice(names)
        if kind == "yn":
            q = ("yn", rng.choice(names), x)
        else:
            q = (kind, x)
        res = answer(q, kp, kr)
        if res is None or res[1] < 2:
            continue
        text = question_text(rel, q)
        if text in used:
            continue
        used.add(text)
        return q, res[0]
    return None


def make_story(rng):
    while True:
        rel, names, parent, roots = new_world(rng)
        facts = [("edge", c, p) for c, p in parent.items()]
        facts += [("root", x) for x in roots]
        rng.shuffle(facts)
        stmts = list(facts)
        for f in facts:
            if len(stmts) >= 12:
                break
            if rng.random() < 0.25:
                stmts.insert(rng.randrange(len(stmts) + 1), f)

        target = rng.randint(1, 5)
        kp, kr = {}, set()
        lines = []
        used = set()
        nq = 0
        for f in stmts:
            if f[0] == "edge":
                kp[f[1]] = f[2]
            else:
                kr.add(f[1])
            lines.append(statement_text(rel, f, rng))
            if nq < target and rng.random() < 0.4:
                got = random_question(rng, names, kp, kr, rel, used)
                if got is not None:
                    q, ans = got
                    lines.append(question_text(rel, q) + "\t" + ans)
                    nq += 1
        for _ in range(60):
            if nq >= target:
                break
            got = random_question(rng, names, kp, kr, rel, used)
            if got is not None:
                q, ans = got
                lines.append(question_text(rel, q) + "\t" + ans)
                nq += 1
        if nq >= 1:
            return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        for i, line in enumerate(make_story(rng), 1):
            out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
