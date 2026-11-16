import random
import sys

# Vocabulary: 8 names, 4 crews, and a handful of fixed words.
PEOPLE = ["ada", "bo", "cy", "dee", "eli", "fay", "gus", "hal"]
CREWS = ["falcons", "ravens", "otters", "lynx"]


def count(events, *names):
    """Number of statements that mention all of the given entities."""
    need = set(names)
    return sum(1 for ents in events if need <= ents)


def apply_statement(members, op):
    kind, p, g, h = op
    if kind == "enlist":
        members[p].add(g)
        return "%s enlists with %s." % (p, g), frozenset([p, g])
    if kind == "withdraw":
        members[p].discard(g)
        return "%s withdraws from %s." % (p, g), frozenset([p, g])
    members[p].discard(g)
    members[p].add(h)
    return "%s transfers from %s to %s." % (p, g, h), frozenset([p, g, h])


def pick_statement(rng, members):
    options = []
    for p in PEOPLE:
        for g in CREWS:
            if g not in members[p]:
                options.append(("enlist", p, g, None))
        for g in sorted(members[p]):
            options.append(("withdraw", p, g, None))
            for h in CREWS:
                if h not in members[p]:
                    options.append(("transfer", p, g, h))
    return rng.choice(options)


def question_candidates(members, events):
    found = []
    for p in PEOPLE:
        if members[p] and count(events, p) >= 2:
            found.append(("Which crews does %s belong to?" % p,
                          ",".join(sorted(members[p]))))
    for g in CREWS:
        if count(events, g) >= 2:
            roster = [p for p in PEOPLE if g in members[p]]
            if roster:
                found.append(("Who belongs to %s?" % g, ",".join(roster)))
            found.append(("What is the headcount of %s?" % g, str(len(roster))))
    for p in PEOPLE:
        for g in CREWS:
            if count(events, p, g) >= 2:
                ans = "yes" if g in members[p] else "no"
                found.append(("Is %s a member of %s?" % (p, g), ans))
    return found


def try_story(rng):
    n_statements = rng.randint(3, 12)
    n_questions = rng.randint(1, 5)
    members = {p: set() for p in PEOPLE}
    events = []
    lines = []
    asked = set()
    statements_left = n_statements
    questions_left = n_questions
    while statements_left or questions_left:
        if questions_left and len(events) >= 2 and (
                not statements_left or rng.random() < 0.4):
            cands = [c for c in question_candidates(members, events)
                     if c[0] not in asked]
            if cands:
                text, answer = rng.choice(cands)
                asked.add(text)
                lines.append("%s?\t%s" % (text.rstrip("?"), answer))
                questions_left -= 1
                continue
        if not statements_left:
            return None  # no valid question here; retry the story
        op = pick_statement(rng, members)
        text, ents = apply_statement(members, op)
        events.append(ents)
        lines.append(text[:1].upper() + text[1:])
        statements_left -= 1
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lines = None
        while lines is None:
            lines = try_story(rng)
        for i, line in enumerate(lines, 1):
            out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
