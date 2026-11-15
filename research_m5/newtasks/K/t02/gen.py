#!/usr/bin/env python3
"""Synthetic exchange-of-things reading-comprehension stories.

Usage: python gen.py N SEED

Prints N stories. People hold things and pass them to each other. Every
statement is applied to a simulated world, and every question is answered
by reading that world at the point where it is asked.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli", "Fay"]
THINGS = ["apple", "book", "coin", "pen", "ball", "cup"]

MIN_STATEMENTS, MAX_STATEMENTS = 3, 12
MIN_QUESTIONS, MAX_QUESTIONS = 1, 5
# A question is only asked about a name or thing that appears in at least
# this many earlier statements, so the answer needs several sentences.
MIN_MENTIONS = 2
QUESTION_PROBABILITY = 0.4


def statement_candidates(holder):
    """Legal statements in the current state.

    Each entry is (text, changes, tokens): the sentence, the new holder of
    each thing it moves, and the names and things it mentions.
    """
    cands = []
    for t in THINGS:
        h = holder[t]
        if h is None:
            for p in NAMES:
                cands.append((f"{p} has the {t}.", {t: p}, {p, t}))
        else:
            for p in NAMES:
                if p != h:
                    cands.append((f"{p} gets the {t} from {h}.", {t: p}, {p, t, h}))
                    cands.append((f"{h} gives the {t} to {p}.", {t: p}, {h, t, p}))
    for p in NAMES:
        for q in NAMES:
            if p == q:
                continue
            for t in THINGS:
                if holder[t] != p:
                    continue
                for u in THINGS:
                    if holder[u] != q:
                        continue
                    cands.append((f"{p} and {q} swap the {t} and the {u}.",
                                  {t: q, u: p}, {p, q, t, u}))
    return cands


def question_candidates(holder, history):
    """Legal questions in the current state, as (text, answer) pairs."""

    def mentions(token):
        return sum(1 for tokens in history if token in tokens)

    cands = []
    for t in THINGS:
        if mentions(t) < MIN_MENTIONS:
            continue
        cands.append((f"Who has the {t}?", holder[t].lower() if holder[t] else "nobody"))
        for p in NAMES:
            cands.append((f"Does {p} have the {t}?", "yes" if holder[t] == p else "no"))
    for p in NAMES:
        if mentions(p) < MIN_MENTIONS:
            continue
        held = sorted(t for t in THINGS if holder[t] == p)
        cands.append((f"What does {p} have?", ",".join(held) if held else "nothing"))
        cands.append((f"How many things does {p} have?", str(len(held))))
    return cands


def try_story(rng, s_target, q_target):
    """Build one story, or return None if it reached a dead end."""
    holder = {t: None for t in THINGS}
    history = []   # one set of names/things per statement so far
    asked = set()
    lines = []
    s_done = q_done = 0

    while s_done < s_target or q_done < q_target:
        s_left = s_target - s_done
        q_left = q_target - q_done

        if q_left and s_done >= 1 and (s_left == 0 or rng.random() < QUESTION_PROBABILITY):
            qs = [c for c in question_candidates(holder, history) if c[0] not in asked]
            if qs:
                text, answer = rng.choice(qs)
                asked.add(text)
                q_done += 1
                lines.append(f"{len(lines) + 1} {text}\t{answer}")
                continue
            if s_left == 0:
                return None

        if s_left == 0:
            return None
        text, changes, tokens = rng.choice(statement_candidates(holder))
        holder.update(changes)
        history.append(tokens)
        s_done += 1
        lines.append(f"{len(lines) + 1} {text}")

    return lines


def make_story(rng):
    while True:
        lines = try_story(rng,
                          rng.randint(MIN_STATEMENTS, MAX_STATEMENTS),
                          rng.randint(MIN_QUESTIONS, MAX_QUESTIONS))
        if lines is not None:
            return lines


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 2
    n = int(argv[1])
    rng = random.Random(int(argv[2]))
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
