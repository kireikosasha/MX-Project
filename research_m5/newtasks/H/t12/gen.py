import random
import sys

# Reasoning kind: ordering events on a week. Each event falls on a day;
# statements relate events to days ("on the day after", "on the day before",
# "on <day>"). Questions ask the day of an event or whether it is on a day.
# Answers come from brute-force enumeration of all day assignments that
# satisfy the statements above the question.

EVENTS = ["party", "picnic", "game", "concert", "quiz", "meeting", "dinner"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def args(s):
    return (s[1],) if s[0] == "on" else (s[1], s[2])


def consistent(stmts, assign):
    for kind, a, b in stmts:
        if kind == "on":
            if a in assign and assign[a] != b:
                return False
        else:
            diff = 1 if kind == "after" else -1
            if a in assign and b in assign and assign[a] != assign[b] + diff:
                return False
    return True


def possible_values(stmts, names, key, limit=2):
    """Set of answers key(assignment) over all assignments consistent with stmts.
    Stops early once limit distinct answers are found."""
    seen = set()
    assign = {}

    def rec(i):
        if i == len(names):
            seen.add(key(assign))
            return
        name = names[i]
        for d in range(len(DAYS)):
            assign[name] = d
            if consistent(stmts, assign):
                rec(i + 1)
                if len(seen) >= limit:
                    del assign[name]
                    return
            del assign[name]

    rec(0)
    return seen


def make_statement(rng, events, hidden):
    for _ in range(40):
        kind = rng.choice(["on", "after", "before"])
        if kind == "on":
            x = rng.choice(events)
            d = hidden[x]
            return ("on", x, d), "The %s is on %s." % (x, DAYS[d])
        x, y = rng.sample(events, 2)
        diff = 1 if kind == "after" else -1
        if hidden[x] == hidden[y] + diff:
            return (kind, x, y), "The %s is on the day %s the %s." % (x, kind, y)
    x = rng.choice(events)
    return ("on", x, hidden[x]), "The %s is on %s." % (x, DAYS[hidden[x]])


def make_question(rng, stmts, names):
    for _ in range(40):
        x = rng.choice(names)
        if rng.random() < 0.5:
            key = lambda a, x=x: DAYS[a[x]]
            text = "What day is the %s?" % x
        else:
            d = rng.randrange(len(DAYS))
            key = lambda a, x=x, d=d: "yes" if a[x] == d else "no"
            text = "Is the %s on %s?" % (x, DAYS[d])
        # The question must not be settled by the statements about x alone,
        # so that it needs several sentences.
        sub = [s for s in stmts if x in args(s)]
        sub_names = sorted({n for s in sub for n in args(s)})
        if len(possible_values(sub, sub_names, key)) == 1:
            continue
        full = possible_values(stmts, names, key)
        if len(full) != 1:
            continue
        return text, full.pop()
    return None


def make_story(rng):
    while True:
        k = rng.randint(3, 5)
        events = rng.sample(EVENTS, k)
        hidden = {e: rng.randrange(len(DAYS)) for e in events}
        n_st = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        slots = ["S"] * n_st + ["Q"] * n_q
        rng.shuffle(slots)
        i = slots.index("S")
        slots[0], slots[i] = slots[i], slots[0]

        stmts = []
        lines = []
        ok = True
        for slot in slots:
            if slot == "S":
                s, text = make_statement(rng, events, hidden)
                stmts.append(s)
                lines.append(text)
            else:
                names = sorted({n for s in stmts for n in args(s)})
                q = make_question(rng, stmts, names)
                if q is None:
                    ok = False
                    break
                lines.append(q)
        if ok:
            return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lines = make_story(rng)
        for i, line in enumerate(lines, 1):
            if isinstance(line, tuple):
                out.append("%d %s\t%s" % (i, line[0], line[1]))
            else:
                out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
