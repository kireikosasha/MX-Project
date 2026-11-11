#!/usr/bin/env python3
"""Synthetic time-ordering reading task.

Stories describe who went to which place on which named day, in mixed
order, sometimes with "the day after <day>". A place is kept until the
next move. Questions ask where someone was on a day, where they went
before or after a day, or whether they were at a place on a day. Answers
are computed from the statements above each question only.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "bob", "cal", "dee", "eli"]
PLACES = ["park", "school", "shop", "lake", "farm"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def make_facts(rng, n_st):
    """Return (facts, texts). A fact is (name, day_index, place)."""
    names = rng.sample(NAMES, rng.randint(3, 5))
    facts = []
    texts = []
    used = set()
    while len(facts) < n_st:
        name = rng.choice(names)
        place = rng.choice(PLACES)
        kind = rng.randrange(3)
        if kind == 2:
            base = rng.randrange(6)  # monday..saturday, so "the day after" stays in the week
            day = base + 1
            text = "%s went to the %s the day after %s." % (name, place, DAYS[base])
        else:
            day = rng.randrange(7)
            if kind == 0:
                text = "%s went to the %s on %s." % (name, place, DAYS[day])
            else:
                text = "on %s %s went to the %s." % (DAYS[day], name, place)
        if (name, day) in used:
            continue  # one move per name per day keeps every answer unique
        used.add((name, day))
        facts.append((name, day, place))
        texts.append(text)
    return facts, texts


def candidate(rng, prefix):
    """Return (question_text_without_mark, answer) answerable from prefix, or None."""
    names = sorted(set(f[0] for f in prefix))
    for _ in range(200):
        name = rng.choice(names)
        day = rng.randrange(7)
        mine = [f for f in prefix if f[0] == name]
        kind = rng.randrange(4)
        if kind == 0:
            prior = [f for f in mine if f[1] <= day]
            if prior:
                ans = max(prior, key=lambda f: f[1])[2]
                return "where was %s on %s" % (name, DAYS[day]), ans
        elif kind == 1:
            prior = [f for f in mine if f[1] < day]
            if prior:
                ans = max(prior, key=lambda f: f[1])[2]
                return "where did %s go before %s" % (name, DAYS[day]), ans
        elif kind == 2:
            later = [f for f in mine if f[1] > day]
            if later:
                ans = min(later, key=lambda f: f[1])[2]
                return "where did %s go after %s" % (name, DAYS[day]), ans
        else:
            prior = [f for f in mine if f[1] <= day]
            if prior:
                place = rng.choice(PLACES)
                cur = max(prior, key=lambda f: f[1])[2]
                ans = "yes" if cur == place else "no"
                return "was %s at the %s on %s" % (name, place, DAYS[day]), ans
    return None


def make_story(rng):
    """Return a list of (kind, text, answer) items; kind is 's' or 'q'."""
    while True:
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        facts, texts = make_facts(rng, n_st)
        items = []
        left = n_q
        for i, text in enumerate(texts):
            items.append(("s", text, None))
            if left > 0 and i >= 1 and rng.random() < 0.5:
                c = candidate(rng, facts[: i + 1])
                if c is not None:
                    items.append(("q", c[0], c[1]))
                    left -= 1
        while left > 0:
            c = candidate(rng, facts)
            if c is None:
                break
            items.append(("q", c[0], c[1]))
            left -= 1
        if left == 0:
            return items


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for k, (kind, text, ans) in enumerate(make_story(rng), 1):
            if kind == "s":
                out.append("%d %s" % (k, text))
            else:
                out.append("%d %s?\t%s" % (k, text, ans))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
