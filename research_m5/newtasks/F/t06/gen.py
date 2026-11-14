"""Synthetic order-in-time reading task generator.

Usage: python gen.py N SEED

Each story is a set of people moving between places at distinct integer
times. Statements are shown in a shuffled order. Questions are asked after
some statements and are answered only from the statements above them.
"""
import random
import sys

NAMES = ["ana", "ben", "cal", "dia", "eli"]
PLACES = ["hall", "lab", "yard", "barn"]
MAXT = 30


def place_at(events, name, cond):
    """Place of the latest event of `name` whose time satisfies cond."""
    best = None
    for e in events:
        if e["person"] == name and cond(e["time"]):
            if best is None or e["time"] > best["time"]:
                best = e
    return best["place"]


def free_times(move_times):
    return [t for t in range(1, MAXT + 1) if t not in move_times]


def candidate(rng, events, persons, move_times):
    """Return (question text, answer, requirement) or None.

    requirement(e) is True for each event whose statement must be known
    for the answer to be determined.
    """
    kind = rng.randrange(5)
    if kind == 0:
        name = rng.choice(persons)
        T = rng.choice(free_times(move_times))
        ans = place_at(events, name, lambda t: t <= T)
        text = f"Where was {name} at time {T}"
        req = lambda e, nm=name, T=T: e["person"] == nm and e["time"] <= T
    elif kind == 1:
        name = rng.choice(persons)
        moves = [e["time"] for e in events
                 if e["person"] == name and e["kind"] == "move"]
        if not moves:
            return None
        T = rng.choice(moves)
        ans = place_at(events, name, lambda t: t < T)
        text = f"Where was {name} before time {T}"
        req = lambda e, nm=name, T=T: e["person"] == nm and e["time"] < T
    elif kind == 2:
        place = rng.choice(PLACES)
        T = rng.choice(free_times(move_times))
        found = [nm for nm in persons
                 if place_at(events, nm, lambda t: t <= T) == place]
        if not found:
            return None
        ans = ",".join(sorted(found))
        text = f"Who was in the {place} at time {T}"
        req = lambda e, T=T: e["time"] <= T
    elif kind == 3:
        name = rng.choice(persons)
        place = rng.choice(PLACES)
        T = rng.randint(1, MAXT + 1)
        done = any(e["person"] == name and e["kind"] == "move"
                   and e["place"] == place and e["time"] < T
                   for e in events)
        ans = "yes" if done else "no"
        text = f"Did {name} go to the {place} before time {T}"
        req = lambda e, nm=name, T=T: e["person"] == nm and e["time"] < T
    else:
        name = rng.choice(persons)
        T = rng.randint(1, MAXT + 1)
        count = sum(1 for e in events
                    if e["person"] == name and e["kind"] == "move"
                    and e["time"] < T)
        ans = str(count)
        text = f"How many moves did {name} make before time {T}"
        req = lambda e, nm=name, T=T: e["person"] == nm and e["time"] < T
    return text, ans, req


def statement(e):
    if e["kind"] == "start":
        return f"At time 0 {e['person']} was in the {e['place']}."
    return f"At time {e['time']} {e['person']} {e['verb']} to the {e['place']}."


def make_story(rng):
    S = rng.randint(4, 12)                      # number of statements
    p = rng.randint(2, min(4, S - 2))           # number of people
    persons = rng.sample(NAMES, p)
    move_times = set(rng.sample(range(1, MAXT + 1), S - p))

    events = []
    cur = {}
    for name in persons:
        cur[name] = rng.choice(PLACES)
        events.append({"person": name, "time": 0, "kind": "start",
                       "verb": "", "place": cur[name]})
    for t in sorted(move_times):
        name = rng.choice(persons)
        cur[name] = rng.choice([x for x in PLACES if x != cur[name]])
        events.append({"person": name, "time": t, "kind": "move",
                       "verb": rng.choice(["went", "walked"]),
                       "place": cur[name]})

    order = list(range(S))
    rng.shuffle(order)

    target = rng.randint(1, 5)
    questions = []                              # (position k, text, answer)
    tries = 0
    while len(questions) < target and tries < 1000:
        tries += 1
        k = rng.randint(1, S)
        known = set(order[:k])
        cand = candidate(rng, events, persons, move_times)
        if cand is None:
            continue
        text, ans, req = cand
        if all(i in known for i, e in enumerate(events) if req(e)):
            questions.append((k, text, ans))

    if not questions:                           # safety fallback
        name = persons[0]
        T = free_times(move_times)[0]
        ans = place_at(events, name, lambda t: t <= T)
        questions.append((S, f"Where was {name} at time {T}", ans))

    lines = []
    for pos, idx in enumerate(order, start=1):
        lines.append(statement(events[idx]))
        for k, text, ans in questions:
            if k == pos:
                lines.append(f"{text}?\t{ans}")
    return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), start=1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
