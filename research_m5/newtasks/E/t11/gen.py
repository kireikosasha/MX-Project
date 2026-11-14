import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eli", "fay"]
KINDS = ["frog", "cat", "bird", "fox"]
TRAITS = ["wet", "loud", "small"]
THINGS = ["boat", "rope", "drum"]
VERBS = ["swim", "climb", "sing"]


def cap(word):
    return word[0].upper() + word[1:]


class World:
    """Simulated world; rules and facts stated so far are the only knowledge."""

    def __init__(self, rng):
        self.names = rng.sample(NAMES, rng.randint(2, 4))
        self.kind_trait = {k: rng.choice(TRAITS) for k in KINDS}
        verbs = VERBS[:]
        rng.shuffle(verbs)
        self.thing_verb = dict(zip(THINGS, verbs))
        self.kind = {}                              # name -> kind
        self.holds = {n: set() for n in self.names}  # name -> things held
        self.trait_rules = set()                    # kinds whose rule was stated
        self.verb_rules = set()                     # things whose rule was stated

    def statement_choices(self):
        choices = []
        for n in self.names:
            if n not in self.kind:
                for k in KINDS:
                    choices.append(("isa", n, k))
            for t in THINGS:
                if t in self.holds[n]:
                    choices.append(("drop", n, t))
                else:
                    choices.append(("has", n, t))
        for k in KINDS:
            if k not in self.trait_rules:
                choices.append(("trait", k, None))
        for t in THINGS:
            if t not in self.verb_rules:
                choices.append(("verb", t, None))
        return choices

    def statement(self, rng):
        op, a, b = rng.choice(self.statement_choices())
        if op == "isa":
            self.kind[a] = b
            return f"{cap(a)} is a {b}."
        if op == "has":
            self.holds[a].add(b)
            return f"{cap(a)} has a {b}."
        if op == "drop":
            self.holds[a].discard(b)
            return f"{cap(a)} drops a {b}."
        if op == "trait":
            self.trait_rules.add(a)
            return f"Every {a} is {self.kind_trait[a]}."
        self.verb_rules.add(a)
        return f"Whoever has a {a} can {self.thing_verb[a]}."

    def question(self, rng):
        qtype = rng.randrange(4)
        if qtype == 0:
            n = rng.choice(self.names)
            t = rng.choice(TRAITS)
            text = f"Is {cap(n)} {t}?"
            k = self.kind.get(n)
            yes = k is not None and k in self.trait_rules and self.kind_trait[k] == t
            ans = "yes" if yes else "no"
        elif qtype == 1:
            n = rng.choice(self.names)
            v = rng.choice(VERBS)
            text = f"Can {cap(n)} {v}?"
            yes = any(t in self.verb_rules and self.thing_verb[t] == v
                      for t in self.holds[n])
            ans = "yes" if yes else "no"
        elif qtype == 2:
            n = rng.choice(self.names)
            text = f"How many things does {cap(n)} have?"
            ans = str(len(self.holds[n]))
        else:
            t = rng.choice(THINGS)
            text = f"Who has a {t}?"
            who = [n for n in self.names if t in self.holds[n]]
            ans = ",".join(sorted(who)) if who else "none"
        return f"{text}\t{ans}"


def make_story(rng):
    world = World(rng)
    n_stat = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_stat + n_q
    qpos = set(rng.sample(range(1, total), n_q))  # first line is a statement
    lines = []
    for i in range(total):
        if i in qpos:
            lines.append(world.question(rng))
        else:
            lines.append(world.statement(rng))
    return [f"{i + 1} {line}" for i, line in enumerate(lines)]


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
