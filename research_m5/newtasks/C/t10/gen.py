"""Synthetic team-membership reading task.

Usage: python gen.py N SEED

People join, leave and move between teams. A person is in at most one team.
Questions are answered by simulating the statements above them in the story.
"""
import random
import sys

NAMES = ["ann", "ben", "cal", "dee", "eli", "fay"]
TEAMS = ["red", "blue", "gold", "green"]


def cap(word):
    return word[0].upper() + word[1:]


def statement_text(kind, person, team):
    if kind == "join":
        return f"{cap(person)} joined the {team} team."
    if kind == "leave":
        return f"{cap(person)} left the {team} team."
    return f"{cap(person)} moved to the {team} team."


def valid_events(state):
    events = []
    for p in NAMES:
        cur = state[p]
        if cur is None:
            for t in TEAMS:
                events.append(("join", p, t))
        else:
            events.append(("leave", p, cur))
            for t in TEAMS:
                if t != cur:
                    events.append(("move", p, t))
    return events


def apply_event(state, kind, p, t):
    if kind == "leave":
        state[p] = None
    else:
        state[p] = t


def candidates(state, history):
    """Return (question, answer) pairs whose evidence is at least two statements."""

    def mentions(entity):
        return sum(1 for ps, ts in history if entity in ps or entity in ts)

    out = []
    for p in NAMES:
        if mentions(p) >= 2:
            out.append((f"Which team is {cap(p)} in?", state[p] or "none"))
    for i, p in enumerate(NAMES):
        for q in NAMES[i + 1:]:
            if mentions(p) >= 1 and mentions(q) >= 1 and mentions(p) + mentions(q) >= 3:
                same = state[p] is not None and state[p] == state[q]
                out.append((f"Are {cap(p)} and {cap(q)} in the same team?",
                            "yes" if same else "no"))
    for t in TEAMS:
        if mentions(t) >= 2:
            members = [p for p in NAMES if state[p] == t]
            out.append((f"Who is in the {t} team?", ",".join(members) or "none"))
            out.append((f"How many people are in the {t} team?", str(len(members))))
    return out


def make_story(rng):
    while True:
        s_count = rng.randint(3, 12)
        q_count = rng.randint(1, 5)
        seq = ["s"] * s_count + ["q"] * q_count
        rng.shuffle(seq)
        if seq[0] != "s" or seq[1] != "s":
            continue

        state = {p: None for p in NAMES}
        history = []   # (persons, teams) named in each statement
        lines = []     # (text, answer or None)
        asked = set()
        ok = True
        for item in seq:
            if item == "s":
                kind, p, t = rng.choice(valid_events(state))
                apply_event(state, kind, p, t)
                history.append(({p}, {t}))
                lines.append((statement_text(kind, p, t), None))
            else:
                cands = [c for c in candidates(state, history) if c[0] not in asked]
                if not cands:
                    ok = False
                    break
                text, ans = rng.choice(cands)
                asked.add(text)
                lines.append((text, ans))
        if ok:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                print(f"{i} {text}")
            else:
                print(f"{i} {text}\t{ans}")


if __name__ == "__main__":
    main()
