import random
import sys

PEOPLE = ["ann", "bob", "cy", "dan", "eve", "fay", "gus"]
THINGS = ["coin", "key", "ball", "map", "pen", "hat"]
CONTAINERS = ["chest", "crate", "tin", "bag", "jar"]
PLACES = ["hall", "kitchen", "garden", "cellar"]
OBJECTS = THINGS + CONTAINERS
TARGETS = CONTAINERS + PLACES


class World:
    def __init__(self, people):
        self.people = people
        self.boss = {}  # person -> boss (a person has at most one boss)
        self.loc = {}   # thing or container -> container or place

    def ancestors(self, p):
        # bosses going upward: [boss, boss of boss, ...]
        out = []
        cur = p
        while cur in self.boss:
            cur = self.boss[cur]
            out.append(cur)
        return out

    def chain(self, obj):
        # locations going upward: [container, ..., place]
        out = []
        cur = obj
        while cur in self.loc:
            cur = self.loc[cur]
            out.append(cur)
        return out

    def contents(self, target):
        # every object that is (directly or indirectly) inside target
        return [o for o in OBJECTS if target in self.chain(o)]


def boss_facts(w):
    # (boss, subordinate) pairs that would change the state without a cycle
    out = []
    for x in w.people:
        for y in w.people:
            if x == y or w.boss.get(y) == x or y in w.ancestors(x):
                continue
            out.append((x, y))
    return out


def loc_facts(w):
    # (object, target) pairs that would change the state without a cycle
    out = []
    for o in OBJECTS:
        for t in TARGETS:
            if o == t or w.loc.get(o) == t or o in w.chain(t):
                continue
            out.append((o, t))
    return out


def cap(s):
    return s[0].upper() + s[1:]


def make_statement(w, rng):
    facts = [("boss", f) for f in boss_facts(w)] + [("loc", f) for f in loc_facts(w)]
    if not facts:
        return None
    kind, (a, b) = rng.choice(facts)
    if kind == "boss":
        w.boss[b] = a
        if rng.random() < 0.5:
            return cap(f"{a} is the boss of {b}.")
        return cap(f"{b} reports to {a}.")
    w.loc[a] = b
    if rng.random() < 0.5:
        return cap(f"the {a} is in the {b}.")
    return cap(f"the {a} is inside the {b}.")


def question_candidates(w, asked):
    cands = []
    for o in OBJECTS:
        ch = w.chain(o)
        if len(ch) >= 2:
            cands.append((f"Where is the {o}?", ch[-1]))
            for t in TARGETS:
                if t == o:
                    continue
                cands.append((f"Is the {o} in the {t}?", "yes" if t in ch else "no"))
    for p in w.people:
        anc = w.ancestors(p)
        if len(anc) >= 2:
            cands.append((f"Who is the boss of the boss of {p}?", anc[1]))
            cands.append((f"How many bosses has {p}?", str(len(anc))))
            for a in w.people:
                if a != p:
                    cands.append((f"Is {a} the boss of the boss of {p}?",
                                  "yes" if anc[1] == a else "no"))
    for t in TARGETS:
        c = w.contents(t)
        if len(c) >= 2:
            cands.append((f"What is in the {t}?", ",".join(sorted(c))))
    return [(q, a) for q, a in cands if q not in asked]


def make_story(rng):
    while True:
        people = rng.sample(PEOPLE, rng.randint(3, 5))
        w = World(people)
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        lines = []
        asked = set()
        st = 0
        q = 0
        failed = False
        while st < n_st or q < n_q:
            if q < n_q and (st == n_st or (st >= 2 and rng.random() < 0.4)):
                cands = question_candidates(w, asked)
                if cands:
                    text, ans = rng.choice(cands)
                    asked.add(text)
                    lines.append(f"{text}\t{ans}")
                    q += 1
                    continue
                if st == n_st:
                    failed = True
                    break
            stmt = make_statement(w, rng)
            if stmt is None:
                failed = True
                break
            lines.append(stmt)
            st += 1
        if not failed:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append(f"{i} {line}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
