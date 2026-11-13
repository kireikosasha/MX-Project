import random
import sys

NAMES = ["ann", "bob", "cy", "dee", "eli", "fay"]
KINDS = ["cat", "dog", "bird", "frog"]
CATS = ["pet", "mammal", "animal"]
THINGS = ["key", "box", "door"]
CAT_RULE_PAIRS = [("pet", "mammal"), ("pet", "animal"), ("mammal", "animal")]
PERM_RULE_PAIRS = [("key", "box"), ("key", "door"), ("box", "door")]


def article(word):
    return "an" if word[0] in "aeiou" else "a"


class World:
    def __init__(self):
        self.kind = {}           # name -> kind
        self.kind_rules = set()  # (kind, category)
        self.cat_rules = set()   # (category, category)
        self.holder = {}         # thing -> name
        self.perm_rules = set()  # (thing, thing)

    def categories(self, name):
        # everything a name is, by following the general rules from its kind
        edges = {}
        for a, b in self.kind_rules | self.cat_rules:
            edges.setdefault(a, set()).add(b)
        start = self.kind[name]
        found = {start}
        stack = [start]
        while stack:
            for nxt in edges.get(stack.pop(), ()):
                if nxt not in found:
                    found.add(nxt)
                    stack.append(nxt)
        return found

    def can_open(self, name, thing):
        return any(t2 == thing and self.holder.get(t1) == name
                   for t1, t2 in self.perm_rules)


def statement(rng, world, names):
    """Apply one random fact or rule to the world; return (sentence, is_rule)."""
    unnamed = [n for n in names if n not in world.kind]
    kind_rules_free = [(k, c) for k in KINDS for c in CATS
                       if (k, c) not in world.kind_rules]
    cat_rules_free = [p for p in CAT_RULE_PAIRS if p not in world.cat_rules]
    perm_free = [p for p in PERM_RULE_PAIRS if p not in world.perm_rules]
    options = []
    if unnamed:
        options.append((3, "kind"))
    if kind_rules_free:
        options.append((2, "kind_rule"))
    if cat_rules_free:
        options.append((1, "cat_rule"))
    if perm_free:
        options.append((1, "perm_rule"))
    options.append((2, "has"))
    if world.holder:
        options.append((2, "gives"))

    pick = rng.randrange(sum(w for w, _ in options))
    for w, opt in options:
        if pick < w:
            break
        pick -= w

    if opt == "kind":
        name = rng.choice(unnamed)
        kind = rng.choice(KINDS)
        world.kind[name] = kind
        return f"{name} is {article(kind)} {kind}.", False
    if opt == "kind_rule":
        k, c = rng.choice(kind_rules_free)
        world.kind_rules.add((k, c))
        return f"every {k} is {article(c)} {c}.", True
    if opt == "cat_rule":
        a, b = rng.choice(cat_rules_free)
        world.cat_rules.add((a, b))
        return f"every {a} is {article(b)} {b}.", True
    if opt == "perm_rule":
        a, b = rng.choice(perm_free)
        world.perm_rules.add((a, b))
        return f"whoever has the {a} can open the {b}.", True
    if opt == "has":
        thing = rng.choice(THINGS)
        name = rng.choice(names)
        world.holder[thing] = name
        return f"{name} has the {thing}.", False
    # gives: the current holder hands the thing to someone else
    thing = rng.choice(sorted(world.holder))
    giver = world.holder[thing]
    receiver = rng.choice([n for n in names if n != giver])
    world.holder[thing] = receiver
    return f"{giver} gives the {thing} to {receiver}.", False


def question(rng, world, names):
    """Ask about the world as it stands; return (question, answer)."""
    known = [n for n in names if n in world.kind]
    options = ["can"]
    if known:
        options += ["is", "what"]
    if world.holder:
        options.append("who")
    kind = rng.choice(options)
    if kind == "is":
        name = rng.choice(known)
        cat = rng.choice(KINDS + CATS)
        answer = "yes" if cat in world.categories(name) else "no"
        return f"is {name} {article(cat)} {cat}?", answer
    if kind == "what":
        name = rng.choice(known)
        return f"what is {name}?", ",".join(sorted(world.categories(name)))
    if kind == "who":
        thing = rng.choice(sorted(world.holder))
        return f"who has the {thing}?", world.holder[thing]
    name = rng.choice(names)
    thing = rng.choice(THINGS)
    answer = "yes" if world.can_open(name, thing) else "no"
    return f"can {name} open the {thing}?", answer


def build_story(rng):
    while True:
        world = World()
        names = rng.sample(NAMES, rng.randint(3, 5))
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        total = n_st + n_q
        q_slots = set(rng.sample(range(1, total), n_q))  # never first line
        items = []
        n_rules = n_facts = 0
        for i in range(total):
            if i in q_slots:
                items.append(question(rng, world, names))
            else:
                text, is_rule = statement(rng, world, names)
                items.append((text, None))
                if is_rule:
                    n_rules += 1
                else:
                    n_facts += 1
        if n_rules and n_facts:
            return items


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        for num, (text, answer) in enumerate(build_story(rng), 1):
            if answer is None:
                out.append(f"{num} {text}")
            else:
                out.append(f"{num} {text}\t{answer}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
