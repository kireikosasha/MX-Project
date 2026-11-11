#!/usr/bin/env python3
"""Synthetic quantity-tracking reading comprehension generator.

Usage: python gen.py N SEED

Prints N stories to standard output. Each statement line is
"<i> <Sentence>." and each question line is "<i> <Question>?<TAB><answer>".
Numbering restarts at 1 for every story, which marks the story boundaries.
Answers come from simulating the holdings of each person.
"""
import random
import sys

NAMES = ["Mary", "John", "Sandra", "Tom", "Lily"]
# (singular, plural) pairs of things that people can hold
THINGS = [("apple", "apples"), ("coin", "coins"), ("pen", "pens"),
          ("ball", "balls"), ("book", "books")]


def count_phrase(n, thing):
    sing, plur = thing
    return f"{n} {sing if n == 1 else plur}"


def make_story(rng):
    while True:
        story = try_story(rng)
        if story is not None:
            return story


def try_story(rng):
    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    names = rng.sample(NAMES, rng.randint(2, 4))
    tidx = rng.sample(range(len(THINGS)), rng.randint(1, 3))
    hold = {(p, t): 0 for p in names for t in tidx}
    stmts = []    # for each statement: the (name, thing index) keys it changes
    lines = []    # (sentence, answer or None)
    n_done = 0
    q_done = 0
    asked = set()

    # Opening possessions: "X has N things." for a few (name, thing) pairs.
    pairs = list(hold.keys())
    rng.shuffle(pairs)
    n_open = rng.randint(1, min(3, len(pairs), n_stmt))
    for p, t in pairs[:n_open]:
        v = rng.randint(1, 5)
        hold[(p, t)] = v
        lines.append((f"{p} has {count_phrase(v, THINGS[t])}.", None))
        stmts.append([(p, t)])
        n_done += 1

    attempts = 0
    while n_done < n_stmt or q_done < n_q:
        attempts += 1
        if attempts > 300:
            return None
        want_q = (q_done < n_q and n_done >= 2
                  and (n_done >= n_stmt or rng.random() < 0.35))
        if want_q:
            q = make_question(rng, names, tidx, hold, stmts)
            if q is not None and q[0] not in asked:
                asked.add(q[0])
                lines.append((q[0] + "?", q[1]))
                q_done += 1
                continue
        if n_done < n_stmt:
            sent, changes = make_statement(rng, names, hold, tidx)
            lines.append((sent, None))
            stmts.append(changes)
            n_done += 1
    return lines


def make_statement(rng, names, hold, tidx):
    while True:
        op = rng.choice(["buys", "finds", "loses", "gives", "gets"])
        t = rng.choice(tidx)
        p = rng.choice(names)
        if op in ("buys", "finds"):
            n = rng.randint(1, 4)
            hold[(p, t)] += n
            return f"{p} {op} {count_phrase(n, THINGS[t])}.", [(p, t)]
        if op == "loses":
            if hold[(p, t)] == 0:
                continue
            n = rng.randint(1, hold[(p, t)])
            hold[(p, t)] -= n
            return f"{p} loses {count_phrase(n, THINGS[t])}.", [(p, t)]
        # Transfer between p (giver) and r (receiver).
        if hold[(p, t)] == 0:
            continue
        r = rng.choice([q for q in names if q != p])
        n = rng.randint(1, hold[(p, t)])
        hold[(p, t)] -= n
        hold[(r, t)] += n
        phrase = count_phrase(n, THINGS[t])
        if op == "gives":
            sent = f"{p} gives {phrase} to {r}."
        else:
            sent = f"{r} gets {phrase} from {p}."
        return sent, [(p, t), (r, t)]


def make_question(rng, names, tidx, hold, stmts):
    kind = rng.choice(["count", "yesno", "most"])
    t = rng.choice(tidx)
    plur = THINGS[t][1]

    if kind == "most":
        # Needs at least two statements about this thing and a unique top holder.
        rel = sum(1 for ch in stmts if any(k[1] == t for k in ch))
        if rel < 2:
            return None
        vals = [hold[(q, t)] for q in names]
        m = max(vals)
        if m == 0 or vals.count(m) != 1:
            return None
        winner = names[vals.index(m)]
        return (f"Who has the most {plur} now", winner.lower())

    p = rng.choice(names)
    # Needs at least two statements that changed this person's holdings.
    rel = sum(1 for ch in stmts if (p, t) in ch)
    if rel < 2:
        return None
    c = hold[(p, t)]
    if kind == "count":
        return (f"How many {plur} does {p} have now", str(c))

    # yes/no: ask about the true count or a neighbouring number.
    if rng.random() < 0.5 or c == 0:
        n = c if rng.random() < 0.5 else c + 1
    else:
        n = c + rng.choice([-1, 1])
    ans = "yes" if n == c else "no"
    return (f"Does {p} have {count_phrase(n, THINGS[t])} now", ans)


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        story = make_story(rng)
        for i, (sent, ans) in enumerate(story, 1):
            if ans is None:
                out.append(f"{i} {sent}")
            else:
                out.append(f"{i} {sent}\t{ans}")
    if out:
        sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
