import random
import sys

# Task: small piles of shells or pebbles move between people.
# Questions ask for a person's running total, a threshold check,
# a comparison between two people, or a list of people with nothing.

NAMES = ["mira", "tobi", "oren", "sela", "dax"]
ITEMS = {"shell": "shells", "pebble": "pebbles"}
CAP = 6  # no pile ever exceeds this, keeps numbers in single digits


def cap(word):
    return word.capitalize()


def item_word(item, n):
    return item if n == 1 else ITEMS[item]


def make_statement(rng, item, counts, touches):
    """Return (sentence, names involved) or None if no move is possible."""
    kinds = ["gain", "gain", "move", "drop"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "gain":
            open_names = [p for p in NAMES if counts[p] < CAP]
            if not open_names:
                continue
            a = rng.choice(open_names)
            n = rng.randint(1, CAP - counts[a])
            counts[a] += n
            touches[a] += 1
            pattern = rng.choice(["{a} carries {n} {it}.", "{a} pockets {n} {it}."])
            text = pattern.format(a=cap(a), n=n, it=item_word(item, n))
            return text, [a]
        if kind == "move":
            senders = [p for p in NAMES if counts[p] > 0]
            rng.shuffle(senders)
            for a in senders:
                receivers = [p for p in NAMES if p != a and counts[p] < CAP]
                if not receivers:
                    continue
                b = rng.choice(receivers)
                n = rng.randint(1, min(counts[a], CAP - counts[b]))
                counts[a] -= n
                counts[b] += n
                touches[a] += 1
                touches[b] += 1
                text = "{a} hands {n} {it} to {b}.".format(
                    a=cap(a), n=n, it=item_word(item, n), b=cap(b))
                return text, [a, b]
            continue
        if kind == "drop":
            senders = [p for p in NAMES if counts[p] > 0]
            if not senders:
                continue
            a = rng.choice(senders)
            n = rng.randint(1, counts[a])
            counts[a] -= n
            touches[a] += 1
            text = "{a} tosses {n} {it} away.".format(
                a=cap(a), n=n, it=item_word(item, n))
            return text, [a]
    return None


def make_question(rng, item, counts, touches, mentioned):
    """Return (question, answer) computed from the state so far, or None."""
    pl = ITEMS[item]
    kinds = ["total", "threshold", "compare", "nobody"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind in ("total", "threshold"):
            cands = [p for p in mentioned if touches[p] >= 2]
            if not cands:
                continue
            a = rng.choice(cands)
            if kind == "total":
                q = "What is the sum of {pl} with {a}?".format(pl=pl, a=cap(a))
                return q, str(counts[a])
            k = rng.randint(1, 5)
            ans = "yes" if counts[a] > k else "no"
            q = "Is the sum of {pl} with {a} above {k}?".format(pl=pl, a=cap(a), k=k)
            return q, ans
        if kind == "compare":
            pairs = [(x, y) for i, x in enumerate(mentioned)
                     for y in mentioned[i + 1:] if counts[x] != counts[y]]
            if not pairs:
                continue
            x, y = rng.choice(pairs)
            if rng.random() < 0.5:
                x, y = y, x
            winner = x if counts[x] > counts[y] else y
            q = "Between {x} and {y}, who carries more {pl}?".format(
                x=cap(x), y=cap(y), pl=pl)
            return q, winner
        if kind == "nobody":
            zeros = sorted(p for p in mentioned if counts[p] == 0)
            if not zeros:
                continue
            q = "Who is without {pl}?".format(pl=pl)
            return q, ",".join(zeros)
    return None


def try_story(rng):
    item = rng.choice(sorted(ITEMS))
    s_count = rng.randint(3, 12)
    q_count = rng.randint(1, 5)
    rest = ["S"] * (s_count - 2) + ["Q"] * q_count
    rng.shuffle(rest)
    plan = ["S", "S"] + rest  # first two lines are always statements

    counts = {p: 0 for p in NAMES}
    touches = {p: 0 for p in NAMES}
    mentioned = []
    lines = []
    for step in plan:
        if step == "S":
            res = make_statement(rng, item, counts, touches)
            if res is None:
                return None
            text, names = res
            lines.append(text)
            for p in names:
                if p not in mentioned:
                    mentioned.append(p)
        else:
            res = make_question(rng, item, counts, touches, mentioned)
            if res is None:
                return None
            q, ans = res
            lines.append(q + "\t" + ans)
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append("{} {}".format(i, line))
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
