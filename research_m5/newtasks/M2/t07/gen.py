import random
import sys

NAMES = ["ann", "bob", "cal", "eve", "fay", "gus"]
ITEMS = [("apple", "apples"), ("pear", "pears"), ("coin", "coins"), ("ball", "balls")]


def cap(word):
    return word[0].upper() + word[1:]


def statement_event(rng, state, intro, noun):
    """Gain, lose, give or take items; updates state and returns the sentence."""
    givers = [p for p in intro if state[p] > 0]
    op = rng.choice(["give", "take", "gain", "lose"])
    if op in ("give", "lose", "take") and not givers:
        op = "gain"
    if op == "give":
        a = rng.choice(givers)
        b = rng.choice([p for p in intro if p != a])
        n = rng.randint(1, state[a])
        state[a] -= n
        state[b] += n
        return f"{cap(a)} gives {n} {noun(n)} to {cap(b)}."
    if op == "take":
        b = rng.choice(givers)
        a = rng.choice([p for p in intro if p != b])
        n = rng.randint(1, state[b])
        state[b] -= n
        state[a] += n
        return f"{cap(a)} takes {n} {noun(n)} from {cap(b)}."
    if op == "lose":
        a = rng.choice(givers)
        n = rng.randint(1, state[a])
        state[a] -= n
        return f"{cap(a)} loses {n} {noun(n)}."
    a = rng.choice(intro)
    n = rng.randint(1, 3)
    state[a] += n
    return f"{cap(a)} gets {n} more {noun(n)}."


def make_question(rng, state, intro, plural):
    """Return (question text, answer) computed from the current state."""
    counts = [state[p] for p in intro]
    top = max(counts)
    kinds = ["count"]
    if len(intro) >= 2:
        kinds += ["compare", "total"]
    if counts.count(top) == 1:
        kinds.append("most")
    kind = rng.choice(kinds)
    if kind == "count":
        a = rng.choice(intro)
        return f"How many {plural} does {cap(a)} have?", str(state[a])
    if kind == "most":
        winner = next(p for p in intro if state[p] == top)
        return f"Who has the most {plural}?", winner
    a, b = rng.sample(intro, 2)
    if kind == "compare":
        ans = "yes" if state[a] > state[b] else "no"
        return f"Does {cap(a)} have more {plural} than {cap(b)}?", ans
    total = state[a] + state[b]
    return f"How many {plural} do {cap(a)} and {cap(b)} have together?", str(total)


def make_story(rng):
    singular, plural = rng.choice(ITEMS)

    def noun(n):
        return singular if n == 1 else plural

    n_stmts = rng.randint(3, 12)
    n_qs = rng.randint(1, 5)
    n_names = rng.randint(2, min(4, n_stmts))
    names = rng.sample(NAMES, n_names)

    state = {}
    intro = []
    stmts = []
    snaps = [None]
    for step in range(n_stmts):
        remaining = n_stmts - step
        need = n_names - len(intro)
        if len(intro) < 2 or (need > 0 and (need == remaining or rng.random() < 0.5)):
            p = names[len(intro)]
            n = rng.randint(0, 5)
            state[p] = n
            intro.append(p)
            stmts.append(f"{cap(p)} has {n} {noun(n)}.")
        else:
            stmts.append(statement_event(rng, state, intro, noun))
        snaps.append((dict(state), list(intro)))

    slots = []
    for _ in range(n_qs):
        k = rng.randint(1, n_stmts)
        st, intro_k = snaps[k]
        text, ans = make_question(rng, st, intro_k, plural)
        slots.append((k, text, ans))
    slots.sort(key=lambda s: s[0])

    lines = []
    qi = 0
    for k in range(1, n_stmts + 1):
        lines.append(stmts[k - 1])
        while qi < len(slots) and slots[qi][0] == k:
            _, text, ans = slots[qi]
            lines.append(f"{text}\t{ans}")
            qi += 1
    return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            print(f"{i} {line}")


if __name__ == "__main__":
    main()
