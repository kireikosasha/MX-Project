import random
import sys

# World: people hold countable items. Facts change the counts, and questions
# ask for the current count, a yes/no comparison, or the holder of the most.
NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli", "Fay"]
OBJS = ["apples", "coins", "stones", "pens", "shells"]
AMOUNTS = [2, 3, 4, 5]  # never 1 in statements, so plural nouns are always right


def make_statement(rng, have, mentioned, first):
    """Apply one random valid fact to the world and return its sentence."""
    while True:
        act = "has" if first else rng.choice(["has", "finds", "gets", "gives", "loses"])
        n = rng.choice(AMOUNTS)
        if act == "has":
            # 'has' only fixes a starting amount for a pair not yet touched.
            fresh = [(a, o) for a in NAMES for o in OBJS if (a, o) not in have]
            if not fresh:
                continue
            a, o = rng.choice(fresh)
            have[(a, o)] = n
            people = [a]
            text = "%s has %d %s." % (a, n, o)
        elif act == "finds":
            a = rng.choice(NAMES)
            o = rng.choice(OBJS)
            have[(a, o)] = have.get((a, o), 0) + n
            people = [a]
            text = "%s finds %d %s." % (a, n, o)
        elif act == "loses":
            a = rng.choice(NAMES)
            o = rng.choice(OBJS)
            if have.get((a, o), 0) < n:
                continue
            have[(a, o)] -= n
            people = [a]
            text = "%s loses %d %s." % (a, n, o)
        else:
            giver, taker = rng.sample(NAMES, 2)
            o = rng.choice(OBJS)
            if have.get((giver, o), 0) < n:
                continue
            have[(giver, o)] -= n
            have[(taker, o)] = have.get((taker, o), 0) + n
            if act == "gets":
                # taker gets from giver; written from the taker's side
                text = "%s gets %d %s from %s." % (taker, n, o, giver)
                people = [taker, giver]
            else:
                text = "%s gives %d %s to %s." % (giver, n, o, taker)
                people = [giver, taker]
        for p in people:
            if p not in mentioned:
                mentioned.append(p)
        return text


def make_question(rng, have, mentioned):
    """Ask a question answerable from the statements so far; return 'text?<TAB>answer'."""
    kind = rng.choice(["count", "more", "most"])
    pairs = list(have)  # every (name, thing) pair touched so far, in order
    if kind == "most":
        o = rng.choice(sorted({p[1] for p in pairs}))
        counts = {a: have.get((a, o), 0) for a in mentioned}
        top = max(counts.values())
        leaders = [a for a in mentioned if counts[a] == top]
        if len(leaders) == 1:
            return "Who has the most %s?\t%s" % (o, leaders[0].lower())
        kind = "count"  # tie: ask something with one answer instead
    if kind == "more":
        a, o = rng.choice(pairs)
        k = rng.choice(AMOUNTS)
        ans = "yes" if have.get((a, o), 0) > k else "no"
        return "Does %s have more than %d %s?\t%s" % (a, k, o, ans)
    a, o = rng.choice(pairs)
    return "How many %s does %s have?\t%d" % (o, a, have.get((a, o), 0))


def build_story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    have = {}
    mentioned = []
    lines = []
    st_left, q_left = n_st, n_q
    while st_left or q_left:
        if lines and q_left and (st_left == 0 or rng.random() < 0.4):
            lines.append(make_question(rng, have, mentioned))
            q_left -= 1
        else:
            lines.append(make_statement(rng, have, mentioned, first=not lines))
            st_left -= 1
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(build_story(rng), 1):
            out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
