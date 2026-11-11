#!/usr/bin/env python3
"""Generate synthetic group-membership reading-comprehension stories.

Usage: python gen.py N SEED

Each story is a sequence of simple statements about people joining, leaving
and moving between teams, interleaved with questions:
  - "In which team is X?" / "Which team is X in?"  -> a team colour or "none"
  - "Are X and Y in the same team?" / "Is X in the same team as Y?" -> yes / no

The answers come from simulating the statements above each question.
A person belongs to at most one team at a time.
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cal", "Dan", "Eve", "Fay", "Gus", "Hal"]
COLORS = ["red", "blue", "green", "gold"]

JOIN_PATTERNS = ["{p} joins the {c} team.", "{p} enters the {c} team."]
LEAVE_PATTERNS = ["{p} leaves the {c} team.", "{p} quits the {c} team."]
MOVE_PATTERNS = ["{p} moves from the {a} team to the {b} team."]
WHICH_PATTERNS = ["In which team is {p}?", "Which team is {p} in?"]
SAME_PATTERNS = ["Are {a} and {b} in the same team?",
                 "Is {a} in the same team as {b}?"]


class World:
    def __init__(self, people):
        self.people = people
        self.team = {p: None for p in people}   # current team or None
        self.events = {p: 0 for p in people}    # number of statements about p

    def random_statement(self, rng):
        free = [p for p in self.people if self.team[p] is None]
        busy = [p for p in self.people if self.team[p] is not None]
        kinds, weights = [], []
        if free:
            kinds.append("join"); weights.append(0.4)
        if busy:
            kinds.append("leave"); weights.append(0.25)
            kinds.append("move"); weights.append(0.35)
        kind = rng.choices(kinds, weights=weights)[0]

        if kind == "join":
            p = rng.choice(free)
            c = rng.choice(COLORS)
            text = rng.choice(JOIN_PATTERNS).format(p=p, c=c)
            self.team[p] = c
        elif kind == "leave":
            p = rng.choice(busy)
            c = self.team[p]
            text = rng.choice(LEAVE_PATTERNS).format(p=p, c=c)
            self.team[p] = None
        else:
            p = rng.choice(busy)
            a = self.team[p]
            b = rng.choice([x for x in COLORS if x != a])
            text = rng.choice(MOVE_PATTERNS).format(p=p, a=a, b=b)
            self.team[p] = b
        self.events[p] += 1
        return text

    def random_question(self, rng):
        """Return 'question?<TAB>answer' or None if no valid question exists now.

        Questions only use people with enough prior statements, so the answer
        needs several sentences of tracking, not a single lookup.
        """
        which_ok = [p for p in self.people if self.events[p] >= 2]
        active = [p for p in self.people if self.events[p] >= 1]
        types = []
        if which_ok:
            types.append("which")
        if len(active) >= 2:
            types.append("same")
        if not types:
            return None
        if rng.choice(types) == "which":
            p = rng.choice(which_ok)
            text = rng.choice(WHICH_PATTERNS).format(p=p)
            ans = self.team[p] if self.team[p] is not None else "none"
        else:
            a, b = rng.sample(active, 2)
            text = rng.choice(SAME_PATTERNS).format(a=a, b=b)
            same = self.team[a] is not None and self.team[a] == self.team[b]
            ans = "yes" if same else "no"
        return text + "\t" + ans


def make_story(rng):
    while True:
        people = rng.sample(NAMES, rng.randint(2, 5))
        world = World(people)
        s_target = rng.randint(3, 12)
        q_target = rng.randint(1, 5)
        items = []
        n_stmt = n_q = 0
        while n_stmt < s_target:
            if n_q < q_target and rng.random() < 0.35:
                q = world.random_question(rng)
                if q is not None:
                    items.append(q)
                    n_q += 1
                    continue
            items.append(world.random_statement(rng))
            n_stmt += 1
        # Optionally add trailing questions at the end of the story.
        for _ in range(3):
            if n_q >= q_target:
                break
            if rng.random() < 0.5:
                q = world.random_question(rng)
                if q is not None:
                    items.append(q)
                    n_q += 1
        if n_q == 0:
            continue
        return items


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python gen.py N SEED")
    n = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    lines = []
    for _ in range(n):
        for i, item in enumerate(make_story(rng), 1):
            lines.append(f"{i} {item}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
