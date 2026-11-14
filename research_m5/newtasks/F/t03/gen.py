"""Synthetic comparison-on-a-scale reading task.

Usage: python gen.py N SEED

Each story is a set of true comparisons between people on one scale
(height or age). Questions are only asked when the answer follows from the
statements above them, and yes/no questions are only asked about pairs that
are not stated directly, so the answer needs a chain of comparisons.
"""
import random
import sys

NAMES = ["ann", "bob", "cal", "dan", "eva", "fay", "gus", "hal"]

# (bigger, smaller, biggest, smallest) words for each scale
SCALES = [
    ("taller", "shorter", "tallest", "shortest"),
    ("older", "younger", "oldest", "youngest"),
]


def closure(facts):
    """Transitive closure of (x, y) pairs meaning x is bigger than y."""
    rel = set(facts)
    while True:
        new = set()
        for a, b in rel:
            for c, d in rel:
                if b == c and a != d and (a, d) not in rel:
                    new.add((a, d))
        if not new:
            return rel
        rel |= new


def capitalize_first(text):
    return text[0].upper() + text[1:]


def candidate_questions(closed, facts, mentioned, big, small, top, bottom):
    """Return (question, answer) pairs that the statements so far determine."""
    out = []
    for x, y in closed:
        if (x, y) in facts or (y, x) in facts:
            continue  # stated directly, so no chaining is needed
        out.append(("Is %s %s than %s?" % (x, big, y), "yes"))
        out.append(("Is %s %s than %s?" % (x, small, y), "no"))
        out.append(("Is %s %s than %s?" % (y, small, x), "yes"))
        out.append(("Is %s %s than %s?" % (y, big, x), "no"))
    if len(mentioned) >= 3:
        for e in mentioned:
            others = [f for f in mentioned if f != e]
            if all((e, f) in closed for f in others):
                out.append(("Who is the %s?" % top, e))
            if all((f, e) in closed for f in others):
                out.append(("Who is the %s?" % bottom, e))
    return out


def try_story(rng):
    k = rng.randint(3, 5)
    ents = rng.sample(NAMES, k)
    values = rng.sample(range(1, 100), k)
    truth = dict(zip(ents, values))  # higher value = bigger on the scale
    big, small, top, bottom = rng.choice(SCALES)
    target_st = rng.randint(3, 12)
    target_q = rng.randint(1, 5)

    facts = set()
    closed = set()
    mentioned = []
    asked = set()
    lines = []
    st = 0
    qs = 0

    while st < target_st or qs < target_q:
        if qs < target_q and st >= 1 and (st >= target_st or rng.random() < 0.35):
            cands = [
                c for c in candidate_questions(
                    closed, facts, mentioned, big, small, top, bottom)
                if c[0] not in asked
            ]
            if cands:
                text, ans = rng.choice(cands)
                asked.add(text)
                lines.append(text + "\t" + ans)
                qs += 1
                continue
            if st >= target_st:
                return None  # cannot place the remaining questions
        # add one more true statement, preferring pairs not yet known
        pairs = [(a, b) for i, a in enumerate(ents) for b in ents[i + 1:]]
        unknown = [(a, b) for a, b in pairs
                   if (a, b) not in closed and (b, a) not in closed]
        a, b = rng.choice(unknown or pairs)
        x, y = (a, b) if truth[a] > truth[b] else (b, a)
        facts.add((x, y))
        closed = closure(facts)
        for e in (x, y):
            if e not in mentioned:
                mentioned.append(e)
        if rng.random() < 0.5:
            text = "%s is %s than %s." % (x, big, y)
        else:
            text = "%s is %s than %s." % (y, small, x)
        lines.append(capitalize_first(text))
        st += 1
    return lines


def make_story(rng):
    while True:
        story = try_story(rng)
        if story is not None:
            return story


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = make_story(rng)
        for i, line in enumerate(story, 1):
            out.append("%d %s" % (i, line))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
