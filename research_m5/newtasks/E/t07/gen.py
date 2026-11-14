import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eli", "fay", "gus", "hal"]
THINGS = ["apples", "coins"]


def cap(word):
    return word[0].upper() + word[1:]


def action(rng, people, hold, thing):
    """Return one statement that changes the holdings (gain, loss, give or take)."""
    rich = [p for p in people if hold[p] >= 2]
    kind = rng.choice(["gain", "lose", "give", "take"])
    if kind == "lose" and rich:
        p = rng.choice(rich)
        n = rng.randint(2, hold[p])
        hold[p] -= n
        return f"{cap(p)} loses {n} {thing}."
    if kind == "give" and rich:
        g = rng.choice(rich)
        r = rng.choice([x for x in people if x != g])
        n = rng.randint(2, hold[g])
        hold[g] -= n
        hold[r] += n
        return f"{cap(g)} gives {n} {thing} to {cap(r)}."
    if kind == "take":
        t = rng.choice(people)
        srcs = [x for x in rich if x != t]
        if srcs:
            s = rng.choice(srcs)
            n = rng.randint(2, hold[s])
            hold[s] -= n
            hold[t] += n
            return f"{cap(t)} takes {n} {thing} from {cap(s)}."
    # gain (also the fallback when no loss, give or take is possible)
    p = rng.choice(people)
    n = rng.randint(2, 9)
    hold[p] += n
    return f"{cap(p)} gets {n} {thing}."


def question(rng, people, hold, thing):
    """Return 'question?<TAB>answer', answered from the current holdings."""
    kind = rng.choice(["count", "who", "yesno"])
    if kind != "count":
        a, b = rng.sample(people, 2)
    if kind == "who" and hold[a] != hold[b]:
        winner = a if hold[a] > hold[b] else b
        return f"Who has more {thing} of {cap(a)} and {cap(b)}?\t{winner}"
    if kind == "yesno":
        ans = "yes" if hold[a] > hold[b] else "no"
        return f"Does {cap(a)} have more {thing} than {cap(b)}?\t{ans}"
    p = rng.choice(people)
    return f"How many {thing} does {cap(p)} have?\t{hold[p]}"


def make_story(rng):
    thing = rng.choice(THINGS)
    people = rng.sample(NAMES, rng.choice([2, 3]))
    hold = {p: rng.randint(2, 9) for p in people}
    s = rng.randint(3, 12)
    q = rng.randint(1, 5)
    entries = [f"{cap(p)} has {hold[p]} {thing}." for p in people]
    items = ["S"] * (s - len(people)) + ["Q"] * q
    rng.shuffle(items)
    for item in items:
        if item == "S":
            entries.append(action(rng, people, hold, thing))
        else:
            entries.append(question(rng, people, hold, thing))
    return entries


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    for _ in range(n):
        for i, entry in enumerate(make_story(rng), 1):
            print(f"{i} {entry}")


if __name__ == "__main__":
    main()
