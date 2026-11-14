#!/usr/bin/env python3
"""Synthetic quantity-tracking reading task (standard library only).

Usage: python gen.py N SEED

Prints N stories. Each story is about a few people who have, get, lose,
give and take a number of one kind of item. Questions ask for the count
someone holds, compare two counts, or name the person with the most.
All answers come from simulating the statements above each question.
"""
import random
import sys

NAMES = ["Sam", "Ann", "Tom", "Lia", "Ben", "Eva"]
ITEMS = [("apple", "apples"), ("coin", "coins"), ("ball", "balls")]


def make_story(rng):
    while True:
        item_sg, item_pl = rng.choice(ITEMS)
        people = rng.sample(NAMES, rng.randint(2, 4))
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)

        counts = {p: 0 for p in people}
        involved = {p: 0 for p in people}   # statements mentioning p so far
        stmts = []
        snaps = []                          # state after each statement

        def word(n):
            return item_sg if n == 1 else item_pl

        def step():
            kind = rng.choice(["gain", "gain", "lose", "give", "take"])
            a, b = rng.sample(people, 2)
            if kind == "gain":
                n = rng.randint(1, 5)
                if involved[a] == 0:
                    text = f"{a} has {n} {word(n)}."
                else:
                    text = f"{a} gets {n} {word(n)}."
                counts[a] += n
                return text, [a]
            if kind == "lose" and counts[a] > 0:
                n = rng.randint(1, min(counts[a], 4))
                counts[a] -= n
                return f"{a} loses {n} {word(n)}.", [a]
            if kind == "give" and counts[a] > 0:
                n = rng.randint(1, min(counts[a], 4))
                counts[a] -= n
                counts[b] += n
                return f"{a} gives {n} {word(n)} to {b}.", [a, b]
            if kind == "take" and counts[b] > 0:
                n = rng.randint(1, min(counts[b], 4))
                counts[b] -= n
                counts[a] += n
                return f"{a} takes {n} {word(n)} from {b}.", [a, b]
            return None

        for _ in range(n_stmt):
            res = None
            while res is None:
                res = step()
            text, names = res
            stmts.append(text)
            for p in set(names):
                involved[p] += 1
            snaps.append((dict(counts), dict(involved)))

        def make_question(snap):
            cnt, inv = snap
            strong = [p for p in people if inv[p] >= 2]   # needs several sentences
            known = [p for p in people if inv[p] >= 1]
            options = []
            for p in strong:
                options.append((f"How many {item_pl} does {p} have?",
                                str(cnt[p])))
            for p in strong:
                for q in strong:
                    if p != q:
                        ans = "yes" if cnt[p] > cnt[q] else "no"
                        options.append(
                            (f"Does {p} have more {item_pl} than {q}?", ans))
            if len(known) >= 2:
                top = max(cnt[p] for p in known)
                winners = [p for p in known if cnt[p] == top]
                if top > 0 and len(winners) == 1:
                    options.append((f"Who has the most {item_pl}?",
                                    winners[0].lower()))
            if not options:
                return None
            return rng.choice(options)

        positions = sorted(rng.sample(range(2, n_stmt + 1),
                                      min(n_q, n_stmt - 1)))
        qmap = {}
        for pos in positions:
            q = make_question(snaps[pos - 1])
            if q is not None:
                qmap[pos] = q
        if not qmap:
            continue

        lines = []
        num = 0
        for j in range(1, n_stmt + 1):
            num += 1
            lines.append(f"{num} {stmts[j - 1]}")
            if j in qmap:
                num += 1
                question, answer = qmap[j]
                lines.append(f"{num} {question}\t{answer}")
        return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n_stories = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    for _ in range(n_stories):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
