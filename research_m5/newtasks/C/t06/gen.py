import random
import sys

# Time-ordering task: people visit places on named days of the week.
# A person stays at their last visited place until their next visit
# (persistence). Statements are told in shuffled order; questions ask where
# someone was on a day, whether they were at a place on a day, or where they
# were just before or just after a visit. A question is kept only if the
# statements above it already give the same answer as the whole story.

NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli", "Fay"]
PLACES = ["park", "market", "library", "harbor", "school"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def visible_place_on(events, name, day):
    best = None
    for n, d, p in events:
        if n == name and d <= day and (best is None or d > best[0]):
            best = (d, p)
    return best[1] if best else None


def find_visit(events, name, place):
    for e in events:
        if e[0] == name and e[2] == place:
            return e
    return None


def neighbour(events, name, day, before):
    if before:
        cands = [e for e in events if e[0] == name and e[1] < day]
        if not cands:
            return None
        return max(cands, key=lambda e: e[1])[2]
    cands = [e for e in events if e[0] == name and e[1] > day]
    if not cands:
        return None
    return min(cands, key=lambda e: e[1])[2]


def question_text(q):
    kind, name, day, place = q
    if kind == "on":
        return "Where was %s on %s?" % (name, DAYS[day])
    if kind == "yn":
        return "Was %s at the %s on %s?" % (name, place, DAYS[day])
    if kind == "before":
        return "Where was %s before the %s?" % (name, place)
    return "Where was %s after the %s?" % (name, place)


def answer(q, events):
    kind, name, day, place = q
    if kind == "on":
        return visible_place_on(events, name, day)
    if kind == "yn":
        here = visible_place_on(events, name, day)
        if here is None:
            return None
        return "yes" if here == place else "no"
    visit = find_visit(events, name, place)
    if visit is None:
        return None
    return neighbour(events, name, visit[1], kind == "before")


def render(rng, e):
    name, d, place = e
    styles = ["on"]
    if d > 0:
        styles.append("after")
    if d < 6:
        styles.append("before")
    style = rng.choice(styles)
    if style == "on":
        return "%s went to the %s on %s." % (name, place, DAYS[d])
    if style == "after":
        return "%s went to the %s the day after %s." % (name, place, DAYS[d - 1])
    return "%s went to the %s the day before %s." % (name, place, DAYS[d + 1])


def try_story(rng):
    people = rng.choice([2, 3])
    names = rng.sample(NAMES, people)
    events = []
    for name in names:
        k = rng.randint(1, 4)
        days = rng.sample(range(7), k)
        places = rng.sample(PLACES, k)
        for d, p in zip(days, places):
            events.append((name, d, p))
    if not 3 <= len(events) <= 12:
        return None

    order = events[:]
    rng.shuffle(order)
    stmts = [render(rng, e) for e in order]
    n = len(order)

    # Candidate questions with their full-story answers.
    cands = []
    for name in names:
        for d in range(7):
            cands.append(("on", name, d, None))
            for p in PLACES:
                cands.append(("yn", name, d, p))
    for e in events:
        cands.append(("before", e[0], None, e[2]))
        cands.append(("after", e[0], None, e[2]))

    pool = []
    for c in cands:
        full = answer(c, events)
        if full is None:
            continue
        for p in range(1, n + 1):
            if answer(c, order[:p]) == full:
                pool.append((p, c, full))
    if not pool:
        return None

    rng.shuffle(pool)
    qn = rng.randint(1, 5)
    chosen = []
    seen = set()
    for p, c, full in pool:
        text = question_text(c)
        if text in seen:
            continue
        seen.add(text)
        chosen.append((p, c, full))
        if len(chosen) == qn:
            break
    chosen.sort(key=lambda x: x[0])

    lines = []
    counter = 0
    qi = 0
    for i in range(1, n + 1):
        counter += 1
        lines.append("%d %s" % (counter, stmts[i - 1]))
        while qi < len(chosen) and chosen[qi][0] == i:
            _, c, full = chosen[qi]
            counter += 1
            lines.append("%d %s\t%s" % (counter, question_text(c), full))
            qi += 1
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(count):
        for line in make_story(rng):
            print(line)


if __name__ == "__main__":
    main()
