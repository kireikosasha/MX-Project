#!/usr/bin/env python3
"""Synthetic time-ordering reading-comprehension stories.

Usage: python gen.py N SEED

People move between places. Each move is told either with its time
("... at time 3.") or relative to another move ("... after / before /
right after ..."), in shuffled order. Every story's statements determine
the timeline uniquely, and questions are answered from that timeline.
"""
import random
import sys
from collections import Counter

NAMES = ["Sam", "Ann", "Tom", "Eve", "Max", "Zoe"]
PLACES = ["park", "kitchen", "garden", "school", "shop"]


def count_timelines(forms, cap=2):
    """Count assignments of times 1..N to the events that satisfy every statement (capped)."""
    n = len(forms)
    full = (1 << n) - 1
    memo = {}

    def allowed(e, mask, last):
        kind, arg = forms[e]
        k = bin(mask).count("1") + 1
        if kind == "at":
            return arg == k
        if kind == "after":
            return ((mask >> arg) & 1) == 1
        if kind == "before":
            return ((mask >> arg) & 1) == 0
        return last == arg  # "right after": arg is the event at time k-1

    def go(mask, last):
        if mask == full:
            return 1
        key = (mask, last)
        if key in memo:
            return memo[key]
        total = 0
        for e in range(n):
            if ((mask >> e) & 1) == 0 and allowed(e, mask, last):
                total += go(mask | (1 << e), e)
                if total >= cap:
                    break
        total = min(total, cap)
        memo[key] = total
        return total

    return go(0, -1)


def make_events(rng):
    n = rng.randint(3, 12)
    where = {}
    events = []
    for t in range(1, n + 1):
        actor = rng.choice(NAMES)
        place = rng.choice([p for p in PLACES if p != where.get(actor)])
        where[actor] = place
        events.append({"actor": actor, "place": place, "time": t})
    return events


def make_forms(rng, events):
    n = len(events)
    pair = Counter((e["actor"], e["place"]) for e in events)
    unique = [j for j, e in enumerate(events) if pair[(e["actor"], e["place"])] == 1]
    forms = []
    for i, e in enumerate(events):
        form = ("at", e["time"])
        cands = [j for j in unique if j != i]
        if cands and rng.random() < 0.5:
            j = rng.choice(cands)
            tj, ti = events[j]["time"], e["time"]
            if tj > ti:
                form = ("before", j)
            elif tj == ti - 1 and rng.random() < 0.5:
                form = ("rafter", j)
            else:
                form = ("after", j)
        forms.append(form)
    # Give explicit times until the timeline is fully determined.
    while count_timelines(forms) != 1:
        i = rng.choice([x for x in range(n) if forms[x][0] != "at"])
        forms[i] = ("at", events[i]["time"])
    return forms, pair


def statement(e, form, events):
    head = f"{e['actor']} went to the {e['place']}"
    kind, arg = form
    if kind == "at":
        return f"{head} at time {arg}."
    j = events[arg]
    ref = f"{j['actor']} went to the {j['place']}"
    word = {"after": "after", "before": "before", "rafter": "right after"}[kind]
    return f"{head} {word} {ref}."


def location(events, actor, t):
    """Place of the latest move by actor at or before time t, or None."""
    best = None
    for e in events:
        if e["actor"] == actor and e["time"] <= t:
            if best is None or e["time"] > best["time"]:
                best = e
    return best


def make_question(rng, events, pair):
    n = len(events)
    unique = [e for e in events if pair[(e["actor"], e["place"])] == 1]
    kind = rng.randrange(5)
    if kind == 0:
        t = rng.randint(1, n)
        actor = rng.choice(NAMES)
        loc = location(events, actor, t)
        if loc is None:
            return None
        return f"Where was {actor} at time {t}?", loc["place"]
    if kind == 1:
        if not unique:
            return None
        ref = rng.choice(unique)
        actor = rng.choice(NAMES)
        loc = location(events, actor, ref["time"] - 1)
        if loc is None:
            return None
        text = f"Where was {actor} before {ref['actor']} went to the {ref['place']}?"
        return text, loc["place"]
    if kind == 2:
        if not unique:
            return None
        e = rng.choice(unique)
        return f"At what time did {e['actor']} go to the {e['place']}?", str(e["time"])
    if kind == 3:
        if len(unique) < 2:
            return None
        a, b = rng.sample(unique, 2)
        ans = "yes" if a["time"] < b["time"] else "no"
        text = (f"Did {a['actor']} go to the {a['place']} before "
                f"{b['actor']} went to the {b['place']}?")
        return text, ans
    t = rng.randint(1, n)
    place = rng.choice(PLACES)
    who = []
    for a in NAMES:
        loc = location(events, a, t)
        if loc is not None and loc["place"] == place:
            who.append(a.lower())
    if not who:
        return None
    return f"Who was in the {place} at time {t}?", ",".join(sorted(who))


def make_story(rng):
    while True:
        events = make_events(rng)
        forms, pair = make_forms(rng, events)
        order = list(range(len(events)))
        rng.shuffle(order)
        lines = [f"{s + 1} {statement(events[i], forms[i], events)}"
                 for s, i in enumerate(order)]
        nq = rng.randint(1, 5)
        questions = []
        seen = set()
        for _ in range(60):
            if len(questions) == nq:
                break
            q = make_question(rng, events, pair)
            if q is not None and q[0] not in seen:
                seen.add(q[0])
                questions.append(q)
        if not questions:
            continue
        for text, ans in questions:
            lines.append(f"{len(lines) + 1} {text}\t{ans}")
        return lines


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    count, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        lines.extend(make_story(rng))
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
