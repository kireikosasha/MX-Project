"""Synthetic containment task: people carry and move things and boxes; places hold everything.

Usage: python gen.py N SEED   (prints N stories; all randomness from random.Random(SEED))
"""
import random
import sys

PERSONS = ["ann", "ben", "cara", "dan"]
THINGS = ["key", "coin", "ball", "book"]
CONTAINERS = ["box", "bag", "tin"]
PLACES = ["hall", "garden", "kitchen", "office"]


class Retry(Exception):
    """A random story could not be finished; the generator starts another one."""


def ref(kind, e):
    if kind[e] == "person":
        return e.capitalize()
    return "the " + e


def cap(s):
    return s[0].upper() + s[1:]


def make_story(rng):
    persons = rng.sample(PERSONS, rng.choice([2, 3]))
    containers = rng.sample(CONTAINERS, rng.choice([1, 2]))
    things = rng.sample(THINGS, rng.choice([2, 3]))
    places = rng.sample(PLACES, rng.choice([3, 4]))

    kind = {}
    for e in persons:
        kind[e] = "person"
    for e in containers:
        kind[e] = "container"
    for e in things:
        kind[e] = "thing"
    for e in places:
        kind[e] = "place"

    parent = {}    # entity -> the place or holder it is directly in
    setidx = {}    # entity -> number of the statement that last set its parent
    pending = sorted(persons + containers + things)   # not introduced yet
    introduced = []
    n_entities = len(pending)
    n_stmts = rng.randint(n_entities, min(12, n_entities + 5))   # always 5..12 here
    n_qs = rng.randint(1, 5)
    asked = set()

    def ancestors(e):
        """Holders above e, ending with the place that finally contains it."""
        out = []
        x = parent[e]
        while kind[x] != "place":
            out.append(x)
            x = parent[x]
        out.append(x)
        return out

    def provenance(e):
        """Statement numbers the current location of e depends on."""
        return {setidx[e]} | {setidx[x] for x in ancestors(e)[:-1]}

    def descendants(y):
        return [x for x in introduced if y in ancestors(x)]

    def intro_options():
        opts = []
        holders = places + [c for c in introduced if kind[c] == "container"]
        for e in pending:
            if kind[e] == "person":
                opts += [("intro", e, p) for p in places]
            else:
                opts += [("intro", e, h) for h in holders]
        return opts

    def action_options():
        opts = []
        people = [x for x in introduced if kind[x] == "person"]
        objs = [x for x in introduced if kind[x] != "person"]
        conts = [x for x in introduced if kind[x] == "container"]
        for p in people:
            loc = parent[p]
            opts += [("goes", p, pl) for pl in places if pl != loc]
            for o in objs:
                if parent[o] == p:
                    for c in conts:
                        if (c != o and o not in ancestors(c) and parent[o] != c
                                and ancestors(c)[-1] == loc
                                and all(x == p or kind[x] != "person"
                                        for x in ancestors(c)[:-1])):
                            opts.append(("puts", p, o, c))
                    for q in people:
                        if q != p and parent[q] == loc:
                            opts.append(("gives", p, o, q))
                elif kind[parent[o]] != "person" and ancestors(o)[-1] == loc:
                    opts.append(("takes", p, o))
        return opts

    def statement_text(o):
        k = o[0]
        if k == "intro":
            return "%s is in %s." % (cap(ref(kind, o[1])), ref(kind, o[2]))
        if k == "goes":
            return "%s goes to %s." % (cap(ref(kind, o[1])), ref(kind, o[2]))
        if k == "takes":
            return "%s takes %s." % (cap(ref(kind, o[1])), ref(kind, o[2]))
        if k == "puts":
            return "%s puts %s in %s." % (cap(ref(kind, o[1])), ref(kind, o[2]),
                                          ref(kind, o[3]))
        return "%s gives %s to %s." % (cap(ref(kind, o[1])), ref(kind, o[2]),
                                       ref(kind, o[3]))

    def apply(o, n):
        k = o[0]
        if k == "intro":
            parent[o[1]] = o[2]
            setidx[o[1]] = n
            pending.remove(o[1])
            introduced.append(o[1])
        elif k == "goes":
            parent[o[1]] = o[2]
            setidx[o[1]] = n
        elif k == "takes":
            parent[o[2]] = o[1]
            setidx[o[2]] = n
        else:   # puts / gives: object o[2] goes into holder o[3]
            parent[o[2]] = o[3]
            setidx[o[2]] = n

    def question_options():
        opts = []
        targets = [x for x in introduced if kind[x] == "container"] + places
        for x in introduced:
            if len(provenance(x)) < 2:
                continue
            opts.append(("where", x))
            for y in targets:
                if y != x:
                    opts.append(("is", x, y))
        for y in targets:
            desc = descendants(y)
            if desc and len({s for d in desc for s in provenance(d)}) >= 2:
                opts.append(("what", y))
        return [o for o in opts if question_text(o) not in asked]

    def question_text(o):
        if o[0] == "where":
            return "Where is %s?" % ref(kind, o[1])
        if o[0] == "is":
            return "Is %s in %s?" % (ref(kind, o[1]), ref(kind, o[2]))
        return "What is in %s?" % ref(kind, o[1])

    def question_answer(o):
        if o[0] == "where":
            return ancestors(o[1])[-1]
        if o[0] == "is":
            return "yes" if o[2] in ancestors(o[1]) else "no"
        return ",".join(sorted(descendants(o[1])))

    lines = []
    stmts = 0
    qs = 0
    while stmts < n_stmts or qs < n_qs:
        left = n_stmts - stmts
        if qs < n_qs and stmts >= 2 and (left == 0 or rng.random() < 0.4):
            opts = question_options()
            if opts:
                o = rng.choice(opts)
                text = question_text(o)
                asked.add(text)
                lines.append("%d %s\t%s" % (len(lines) + 1, text, question_answer(o)))
                qs += 1
                continue
        if left == 0:
            raise Retry()
        intro_opts = intro_options() if pending else []
        act_opts = action_options()
        if intro_opts and len(pending) >= left:
            o = rng.choice(intro_opts)
        elif intro_opts and (not act_opts or rng.random() < 0.35):
            o = rng.choice(intro_opts)
        elif act_opts:
            o = rng.choice(act_opts)
        else:
            raise Retry()
        stmts += 1
        apply(o, stmts)
        lines.append("%d %s" % (len(lines) + 1, statement_text(o)))
    return lines


def make_story_retry(rng):
    for _ in range(100000):
        try:
            return make_story(rng)
        except Retry:
            continue
    raise RuntimeError("could not build a story")


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n_stories):
        for line in make_story_retry(rng):
            print(line)


if __name__ == "__main__":
    main()
