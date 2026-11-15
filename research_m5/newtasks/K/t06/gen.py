"""Synthetic order-in-time reading task generator.

Usage: python gen.py N SEED

Each story is a list of events (person, place, time). A statement either gives
the clock time of its event ("At 5 Ann went to the hall.") or gives its time
relative to an earlier-stated event ("Ben went to the park right after Ann went
to the hall."), so statements are told out of time order. Questions are only
kept when the answer can be derived from the statements above them and is the
same as in the full story.
"""
import random
import sys

NAMES = ["Ann", "Ben", "Cal", "Dee", "Eve", "Fay"]
PLACES = ["kitchen", "garden", "hall", "office", "park", "shed"]
TIMES = list(range(1, 13))


def place_at(evs, name, t, strict=False):
    """Place of `name` at clock time t (or strictly before t if strict)."""
    best = None
    for e in evs:
        if e["name"] != name:
            continue
        ok = (e["time"] < t) if strict else (e["time"] <= t)
        if ok and (best is None or e["time"] > best["time"]):
            best = e
    return None if best is None else best["place"]


def answer(q, evs):
    kind = q["kind"]
    if kind == "where":
        return place_at(evs, q["x"], q["t"])
    if kind == "before":
        target = [e for e in evs if e["name"] == q["y"] and e["place"] == q["p"]]
        if not target:
            return None
        return place_at(evs, q["x"], target[0]["time"], strict=True)
    if kind == "first":
        at = [e for e in evs if e["place"] == q["p"]]
        if len(at) < 2:
            return None
        return min(at, key=lambda e: e["time"])["name"].lower()
    if kind == "yes":
        pl = place_at(evs, q["x"], q["t"])
        if pl is None:
            return None
        return "yes" if pl == q["p"] else "no"
    return None


def make_question(rng, prefix, full):
    kind = rng.choice(["where", "before", "first", "yes"])
    x = rng.choice(NAMES)
    t = rng.choice(TIMES)
    p = rng.choice(PLACES)
    if kind == "where":
        q = {"kind": kind, "x": x, "t": t}
        text = "Where was %s at %d?" % (x, t)
    elif kind == "before":
        e = rng.choice(prefix)
        x = rng.choice([n for n in NAMES if n != e["name"]])
        q = {"kind": kind, "x": x, "y": e["name"], "p": e["place"]}
        text = "Where was %s just before %s went to the %s?" % (x, e["name"], e["place"])
    elif kind == "first":
        p = rng.choice([e["place"] for e in prefix])
        q = {"kind": kind, "p": p}
        text = "Who went to the %s first?" % p
    else:
        q = {"kind": kind, "x": x, "t": t, "p": p}
        text = "Was %s in the %s at %d?" % (x, p, t)
    a = answer(q, prefix)
    if a is None or a != answer(q, full):
        return None
    return text, a


def try_story(rng):
    n = rng.randint(3, 12)
    events = []
    pairs = set()
    times = set()
    stmts = []
    for i in range(n):
        while True:
            name = rng.choice(NAMES)
            place = rng.choice(PLACES)
            if (name, place) not in pairs:
                break
        t = None
        text = None
        if i > 0 and rng.random() < 0.5:
            ref = rng.choice(events)
            side = rng.choice(["after", "before"])
            cand = ref["time"] + 1 if side == "after" else ref["time"] - 1
            if cand in TIMES and cand not in times:
                t = cand
                text = "%s went to the %s right %s %s went to the %s." % (
                    name, place, side, ref["name"], ref["place"])
        if t is None:
            t = rng.choice([x for x in TIMES if x not in times])
            text = "At %d %s went to the %s." % (t, name, place)
        pairs.add((name, place))
        times.add(t)
        events.append({"name": name, "place": place, "time": t})
        stmts.append(text)

    target = rng.randint(1, 5)
    qs = []
    for _ in range(60):
        if len(qs) >= target:
            break
        k = rng.randint(1, n)
        res = make_question(rng, events[:k], events)
        if res:
            qs.append((k, res[0], res[1]))
    if not qs:
        return None
    qs.sort(key=lambda z: z[0])

    lines = []
    qi = 0
    for i, text in enumerate(stmts, 1):
        lines.append("%d %s" % (len(lines) + 1, text))
        while qi < len(qs) and qs[qi][0] == i:
            _, qtext, qans = qs[qi]
            lines.append("%d %s\t%s" % (len(lines) + 1, qtext, qans))
            qi += 1
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines is not None:
            return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
