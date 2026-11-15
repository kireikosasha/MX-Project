#!/usr/bin/env python3
"""Synthetic kinds-and-properties reading task.

Individuals (people) belong to kinds. Kinds form a tree (dog -> mammal -> animal)
and kinds have properties. An individual has the properties of every kind it
belongs to, directly or through the chain of ancestor kinds, plus its own
properties. Questions ask about such derived properties or kind membership.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal"]
KINDS = ["animal", "mammal", "dog", "cat", "fox", "bird", "fish"]
PARENT = {"mammal": "animal", "bird": "animal", "fish": "animal",
          "dog": "mammal", "cat": "mammal", "fox": "mammal"}
PROPS = ["loud", "fast", "calm", "big", "hungry"]


def art(word):
    return "an" if word[0] in "aeiou" else "a"


class World:
    def __init__(self):
        self.texts = set()
        self.count = 0          # number of statements so far
        self.parent = {}        # kind -> (parent kind, statement index)
        self.member = {}        # name -> [(kind, statement index)]
        self.kprop = {}         # kind -> [(prop, statement index)]
        self.iprop = {}         # name -> [(prop, statement index)]

    def add(self, text, fact):
        if text in self.texts:
            return False
        self.texts.add(text)
        idx = self.count
        self.count += 1
        tag, a, b = fact
        if tag == "mem":
            self.member.setdefault(a, []).append((b, idx))
        elif tag == "sub":
            self.parent[a] = (b, idx)
        elif tag == "kprop":
            self.kprop.setdefault(a, []).append((b, idx))
        else:
            self.iprop.setdefault(a, []).append((b, idx))
        return True

    def seen_names(self):
        return sorted(set(self.member) | set(self.iprop))

    def kind_chain(self, kind, mindex):
        """Yield (ancestor, support) for a kind and each of its ancestors."""
        cur, sup = kind, [mindex]
        yield cur, list(sup)
        while cur in self.parent:
            par, idx = self.parent[cur]
            sup = sup + [idx]
            cur = par
            yield cur, list(sup)

    def name_kinds(self, name):
        for kind, midx in self.member.get(name, []):
            yield from self.kind_chain(kind, midx)

    def prop_support(self, name, prop):
        """Shortest list of supporting statements showing name has prop, or None."""
        best = None
        for p, idx in self.iprop.get(name, []):
            if p == prop:
                best = [idx]
        for anc, sup in self.name_kinds(name):
            for p, idx in self.kprop.get(anc, []):
                if p == prop:
                    cand = sup + [idx]
                    if best is None or len(cand) < len(best):
                        best = cand
        return best

    def kind_support(self, name, kind):
        """Shortest list of supporting statements showing name is a kind, or None."""
        best = None
        for anc, sup in self.name_kinds(name):
            if anc == kind and (best is None or len(sup) < len(best)):
                best = sup
        return best


def make_statement(rng, w, story_names):
    """Add one random statement to the world; return its text, or None if not added."""
    r = rng.random()
    if r < 0.35:
        name = rng.choice(story_names)
        kind = rng.choice(KINDS)
        text, fact = f"{name} is {art(kind)} {kind}.", ("mem", name, kind)
    elif r < 0.5:
        free = [c for c in sorted(PARENT) if c not in w.parent]
        if not free:
            return None
        child = rng.choice(free)
        par = PARENT[child]
        text, fact = f"Every {child} is {art(par)} {par}.", ("sub", child, par)
    elif r < 0.75:
        kind = rng.choice(KINDS)
        prop = rng.choice(PROPS)
        text, fact = f"Every {kind} is {prop}.", ("kprop", kind, prop)
    else:
        name = rng.choice(story_names)
        prop = rng.choice(PROPS)
        text, fact = f"{name} is {prop}.", ("iprop", name, prop)
    return text if w.add(text, fact) else None


def make_question(rng, w):
    """Return (question line text with answer, has_reasoning) or None."""
    seen = w.seen_names()
    if not seen:
        return None
    r = rng.random()
    if r < 0.4:
        name = rng.choice(seen)
        prop = rng.choice(PROPS)
        sup = w.prop_support(name, prop)
        text = f"Is {name} {prop}?"
    elif r < 0.8:
        name = rng.choice(seen)
        kind = rng.choice(KINDS)
        sup = w.kind_support(name, kind)
        text = f"Is {name} {art(kind)} {kind}?"
    else:
        prop = rng.choice(PROPS)
        names = [n for n in seen if w.prop_support(n, prop)]
        ans = ",".join(n.lower() for n in names) or "none"
        return f"Which names are {prop}?\t{ans}", False
    if sup:
        if len(sup) < 2:
            return None          # a single sentence is not enough reasoning
        return f"{text}\tyes", True
    if name not in w.member:
        return None              # "no" is only asked about individuals with a known kind
    return f"{text}\tno", False


def try_story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    story_names = rng.sample(NAMES, rng.randint(2, 4))
    w = World()
    lines = []
    st = qn = 0
    reasoned = False
    guard = 0
    while st < n_st or qn < n_q:
        guard += 1
        if guard > 300:
            return None
        if qn < n_q and st >= 2 and (st >= n_st or rng.random() < 0.4):
            res = make_question(rng, w)
            if res is not None:
                line, good = res
                lines.append(line)
                qn += 1
                reasoned = reasoned or good
                continue
            if st >= n_st:
                return None
        if st >= n_st:
            return None
        text = make_statement(rng, w, story_names)
        if text is not None:
            lines.append(text)
            st += 1
    return lines if reasoned and len(w.seen_names()) >= 2 else None


def make_story(rng):
    for _ in range(10000):
        lines = try_story(rng)
        if lines:
            return lines
    raise RuntimeError("could not generate a story")


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
