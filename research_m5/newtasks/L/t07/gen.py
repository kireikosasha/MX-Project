"""Generate synthetic quantity-tracking reading-comprehension stories.

Usage: python gen.py N SEED

People hold counts of things. Statements set, add, remove or transfer counts;
questions ask for one count, for the person with the most, or for a
comparison of two counts. Answers come from simulating the statements above
each question, so they are never written by hand.
"""
import random
import sys

NAMES = ["Sam", "Ann", "Tom", "Lily", "Max", "Ben"]
KINDS = [("apple", "apples"), ("pear", "pears"), ("coin", "coins"), ("ball", "balls")]
MAX_COUNT = 5    # no count ever exceeds this, so answers stay in 0-5
MAX_AMOUNT = 4   # amounts in statements are 1-4


def noun(kind, n):
    return kind[0] if n == 1 else kind[1]


def make_statement(rng, names, kinds, count, touched):
    """Return (text, people_mentioned) and update the state, or None."""
    for _ in range(200):
        act = rng.choice(["has", "gets", "loses", "gives", "takes"])
        kind = rng.choice(kinds)
        a, b = rng.sample(names, 2)
        n = rng.randint(1, MAX_AMOUNT)
        ka, kb = (a, kind), (b, kind)
        ca, cb = count.get(ka, 0), count.get(kb, 0)
        if act == "has":
            # "has" only sets the starting count of a pair not yet touched
            if ka in touched:
                continue
            count[ka] = n
            touched.add(ka)
            return f"{a} has {n} {noun(kind, n)}.", [a]
        if act == "gets":
            if ca + n > MAX_COUNT:
                continue
            count[ka] = ca + n
            touched.add(ka)
            return f"{a} gets {n} more {noun(kind, n)}.", [a]
        if act == "loses":
            if n > ca:
                continue
            count[ka] = ca - n
            touched.add(ka)
            return f"{a} loses {n} {noun(kind, n)}.", [a]
        if act == "gives":
            if n > ca or cb + n > MAX_COUNT:
                continue
            count[ka] = ca - n
            count[kb] = cb + n
            touched.update((ka, kb))
            return f"{a} gives {n} {noun(kind, n)} to {b}.", [a, b]
        # takes: a takes n from b
        if n > cb or ca + n > MAX_COUNT:
            continue
        count[kb] = cb - n
        count[ka] = ca + n
        touched.update((ka, kb))
        return f"{a} takes {n} {noun(kind, n)} from {b}.", [a, b]
    return None


def make_question(rng, kinds, count, mentioned):
    """Return (question_text, answer) from the state so far, or None."""
    kind = rng.choice(kinds)
    plural = kind[1]
    order = ["how", "most", "more"]
    rng.shuffle(order)
    for qt in order:
        if qt == "how":
            a = rng.choice(mentioned)
            n = count.get((a, kind), 0)
            return f"How many {plural} does {a} have?", str(n)
        if qt == "most":
            totals = {m: count.get((m, kind), 0) for m in mentioned}
            best = max(totals.values())
            winners = [m for m in mentioned if totals[m] == best]
            if best > 0 and len(winners) == 1:
                return f"Who has the most {plural}?", winners[0].lower()
        if qt == "more" and len(mentioned) >= 2:
            a, b = rng.sample(mentioned, 2)
            ca, cb = count.get((a, kind), 0), count.get((b, kind), 0)
            if ca != cb:
                return (f"Does {a} have more {plural} than {b}?",
                        ("yes" if ca > cb else "no"))
    return None


def make_story(rng):
    while True:
        names = rng.sample(NAMES, rng.randint(2, 4))
        kinds = rng.sample(KINDS, rng.randint(1, 2))
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        total = n_stmt + n_q
        # line 0 is always a statement, so every question has something above it
        q_pos = set(rng.sample(range(1, total), n_q))
        count, touched, mentioned, lines = {}, set(), [], []
        ok = True
        for t in range(total):
            if t in q_pos:
                res = make_question(rng, kinds, count, mentioned)
                if res is None:
                    ok = False
                    break
                lines.append(res)
            else:
                res = make_statement(rng, names, kinds, count, touched)
                if res is None:
                    ok = False
                    break
                text, people = res
                for p in people:
                    if p not in mentioned:
                        mentioned.append(p)
                lines.append((text, None))
        if ok:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n_stories = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n_stories):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                out.append(f"{i} {text}")
            else:
                out.append(f"{i} {text}\t{ans}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
