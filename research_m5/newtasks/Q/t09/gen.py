import random
import sys

# Small closed vocabulary (about 31 distinct words in total).
NAMES = ["ada", "bo", "cy", "dee", "eli", "fay"]
THINGS = ["coin", "key", "tin", "jar", "chest"]


class Retry(Exception):
    pass


def cap(s):
    return s[0].upper() + s[1:]


def superiors(link, x):
    """Chain of superiors of x, nearest first, top last."""
    chain = []
    while x in link:
        x = link[x]
        chain.append(x)
    return chain


def top_of(link, x):
    chain = superiors(link, x)
    return chain[-1] if chain else x


# ---- facts -------------------------------------------------------------

def obey_options(people, sup):
    """New 'x obeys y' facts that keep a forest (one superior each, no cycle)."""
    out = []
    for x in people:
        if x in sup:
            continue
        for y in people:
            if y != x and x not in superiors(sup, y):
                out.append((x, y))
    return out


def within_options(things, cont):
    """New 'the t sits within the c' facts (one container each, no cycle)."""
    out = []
    for t in things:
        if t in cont:
            continue
        for c in things:
            if c != t and t not in superiors(cont, c):
                out.append((t, c))
    return out


def render_obey(rng, x, y):
    if rng.random() < 0.5:
        return cap(f"{x} obeys {y}.")
    return cap(f"{y} commands {x}.")


def render_within(rng, t, c):
    if rng.random() < 0.5:
        return cap(f"the {t} sits within the {c}.")
    return cap(f"the {c} contains the {t}.")


# ---- questions (answered only from the statements above) ---------------

def question_candidates(people, things, sup, cont):
    out = []
    for x in people:
        for y in people:
            if x == y:
                continue
            chain = superiors(sup, x)
            if y in chain:
                if chain.index(y) >= 1:  # needs two or more links
                    out.append((cap(f"does {x} obey {y}?"), "yes"))
            elif top_of(sup, x) == top_of(sup, y):
                out.append((cap(f"does {x} obey {y}?"), "no"))
        chain = superiors(sup, x)
        if len(chain) >= 2:
            out.append((cap(f"who ranks highest above {x}?"), chain[-1]))
    for y in people:
        under = [x for x in people if y in superiors(sup, x)]
        if under and any(superiors(sup, x).index(y) >= 1 for x in under):
            out.append((cap(f"which names are under {y}?"), ",".join(sorted(under))))
    for t in things:
        for c in things:
            if t == c:
                continue
            chain = superiors(cont, t)
            if c in chain:
                if chain.index(c) >= 1:
                    out.append((cap(f"is the {t} within the {c}?"), "yes"))
            elif top_of(cont, t) == top_of(cont, c):
                out.append((cap(f"is the {t} within the {c}?"), "no"))
        chain = superiors(cont, t)
        if len(chain) >= 2:
            out.append((cap(f"what finally holds the {t}?"), chain[-1]))
    return out


# ---- stories -----------------------------------------------------------

def make_story(rng):
    people = rng.sample(NAMES, rng.randint(3, 6))
    things = rng.sample(THINGS, rng.randint(3, 5))
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_st + n_q
    q_slots = set(rng.sample(range(1, total), n_q))  # never the first line
    sup, cont = {}, {}
    known = []  # facts stated so far: ("o", x, y) or ("c", t, c)
    asked = set()  # questions already asked in this story
    story = []
    for i in range(total):
        if i in q_slots:
            cands = [c for c in question_candidates(people, things, sup, cont)
                     if c[0] not in asked]
            if not cands:
                raise Retry()
            chained = [c for c in cands if c[1] != "no"]
            if chained and rng.random() < 0.75:
                cands = chained  # favour answers that need a chain of links
            q = rng.choice(cands)
            asked.add(q[0])
            story.append(q)
            continue
        moves = [("o", x, y) for x, y in obey_options(people, sup)]
        moves += [("c", t, c) for t, c in within_options(things, cont)]
        if known and (not moves or rng.random() < 0.25):
            kind, a, b = rng.choice(known)  # restate a known fact
        elif moves:
            kind, a, b = rng.choice(moves)
            known.append((kind, a, b))
            if kind == "o":
                sup[a] = b
            else:
                cont[a] = b
        else:
            raise Retry()
        if kind == "o":
            text = render_obey(rng, a, b)
        else:
            text = render_within(rng, a, b)
        story.append((text, None))
    return story


def build_story(rng):
    while True:
        try:
            return make_story(rng)
        except Retry:
            continue


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        story = build_story(rng)
        for i, (text, ans) in enumerate(story, 1):
            if ans is None:
                lines.append(f"{i} {text}")
            else:
                lines.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
