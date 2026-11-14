"""Synthetic reading task: exchange of things.

People hold things; they give, take and swap them. Questions ask who holds
a thing, what a person has, how many things a person has, or whether a
person has a given thing.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "bob", "cal", "dee", "eve", "fay"]
ITEMS = ["ball", "box", "coin", "cup", "hat", "key", "map", "pen"]


class World:
    def __init__(self, rng):
        self.rng = rng
        self.people = rng.sample(NAMES, rng.randint(3, len(NAMES)))
        self.pool = rng.sample(ITEMS, rng.randint(3, len(ITEMS)))
        self.holder = {}     # introduced item -> person who holds it
        self.mentions = {}   # item -> number of statements naming it

    def items_of(self, person):
        return sorted(x for x, h in self.holder.items() if h == person)

    def mention(self, *items):
        for x in items:
            self.mentions[x] = self.mentions.get(x, 0) + 1

    def statement(self):
        rng = self.rng
        fresh = [x for x in self.pool if x not in self.holder]
        holders = [p for p in self.people if self.items_of(p)]
        ops = []
        if fresh:
            ops.append("has")
        if holders:
            ops += ["gives", "takes"]
        if len(holders) >= 2:
            ops.append("swap")
        op = rng.choice(ops)
        if op == "has":
            x = rng.choice(fresh)
            p = rng.choice(self.people)
            self.holder[x] = p
            self.mention(x)
            return f"{p} has a {x}."
        if op == "gives":
            p = rng.choice(holders)
            x = rng.choice(self.items_of(p))
            q = rng.choice([r for r in self.people if r != p])
            self.holder[x] = q
            self.mention(x)
            return f"{p} gives the {x} to {q}."
        if op == "takes":
            q = rng.choice(holders)
            x = rng.choice(self.items_of(q))
            p = rng.choice([r for r in self.people if r != q])
            self.holder[x] = p
            self.mention(x)
            return f"{p} takes the {x} from {q}."
        p = rng.choice(holders)
        q = rng.choice([r for r in holders if r != p])
        x = rng.choice(self.items_of(p))
        y = rng.choice(self.items_of(q))
        self.holder[x], self.holder[y] = q, p
        self.mention(x, y)
        return f"{p} and {q} swap the {x} for the {y}."

    def question(self):
        rng = self.rng
        kinds = ["what", "count"]
        if self.holder:
            kinds += ["who", "does"]
        kind = rng.choice(kinds)
        if kind in ("who", "does"):
            moved = [x for x in self.holder if self.mentions[x] >= 2]
            x = rng.choice(moved or list(self.holder))
            if kind == "who":
                return f"who has the {x}?", self.holder[x]
            p = rng.choice(self.people)
            answer = "yes" if self.holder[x] == p else "no"
            return f"does {p} have the {x}?", answer
        p = rng.choice(self.people)
        mine = self.items_of(p)
        if kind == "what":
            return f"what does {p} have?", ",".join(mine) if mine else "nothing"
        return f"how many things does {p} have?", str(len(mine))


def story(rng):
    w = World(rng)
    n_statements = rng.randint(3, 12)
    n_questions = rng.randint(1, 5)
    total = n_statements + n_questions
    question_at = set(rng.sample(range(1, total), n_questions))
    lines = []
    for i in range(total):
        if i in question_at:
            text, answer = w.question()
            lines.append(f"{i + 1} {text}\t{answer}")
        else:
            lines.append(f"{i + 1} {w.statement()}")
    return lines


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        lines.extend(story(rng))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
