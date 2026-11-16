import itertools
import random
import sys

# Task: a few people land at stops during numbered time slots. The facts are
# given out of order: some with a slot number, some as "one slot after" or
# "before" another arrival. Questions ask about the state during a slot or
# about the order of arrivals. Answers come from a hidden true slot order, and
# a question is only asked when the statements above it fix the answer.

NAMES = ["Ada", "Bo", "Cy", "Dee", "Eli"]
PLACES = ["mill", "barn", "dock", "hall"]
ALL_PAIRS = [(c, p) for c in range(len(NAMES)) for p in range(len(PLACES))]
# PERMS[K]: each tuple pm has pm[e] = slot number of event e (slots 1..K)
PERMS = {k: list(itertools.permutations(range(1, k + 1))) for k in range(3, 7)}


def latest_place(events, pm, c, s, inclusive):
    best_slot, best = 0, None
    for e, (ec, ep) in enumerate(events):
        if ec != c:
            continue
        sl = pm[e]
        ok = sl <= s if inclusive else sl < s
        if ok and sl > best_slot:
            best_slot, best = sl, ep
    return best


def make_question(rng, events, K):
    kind = rng.randrange(5)
    c = rng.randrange(len(NAMES))
    p = rng.randrange(len(PLACES))
    s = rng.randint(1, K)
    if kind == 0:
        text = f"Which place held {NAMES[c]} during slot {s}?"

        def fn(pm):
            r = latest_place(events, pm, c, s, True)
            return None if r is None else PLACES[r]
    elif kind == 1:
        text = f"Was {NAMES[c]} at the {PLACES[p]} during slot {s}?"

        def fn(pm):
            return "yes" if latest_place(events, pm, c, s, True) == p else "no"
    elif kind == 2:
        text = f"Who reached the {PLACES[p]} first?"

        def fn(pm):
            best_slot, best = None, None
            for e, (ec, ep) in enumerate(events):
                if ep == p and (best_slot is None or pm[e] < best_slot):
                    best_slot, best = pm[e], ec
            return None if best is None else NAMES[best].lower()
    elif kind == 3:
        text = f"How many places did {NAMES[c]} reach before slot {s}?"

        def fn(pm):
            return str(sum(1 for e, (ec, _) in enumerate(events)
                           if ec == c and pm[e] < s))
    else:
        text = f"Which place did {NAMES[c]} reach last before slot {s}?"

        def fn(pm):
            r = latest_place(events, pm, c, s, False)
            return None if r is None else PLACES[r]
    return text, fn


def try_question(rng, events, true, cons, K, seen):
    for _ in range(60):
        text, fn = make_question(rng, events, K)
        if text in seen:
            continue
        truth = fn(true)
        if truth is None:
            continue
        # the answer must be the same in every slot order that fits the
        # statements above, so the reader can derive it
        if all(fn(pm) == truth for pm in cons):
            seen.add(text)
            return f"{text}\t{truth}"
    return None


def make_story(rng):
    while True:
        K = rng.randint(3, 6)
        events = rng.sample(ALL_PAIRS, K)
        slots = list(range(1, K + 1))
        rng.shuffle(slots)
        true = tuple(slots)

        def ev_text(e):
            return f"{NAMES[events[e][0]]} reached the {PLACES[events[e][1]]}"

        def timed(e):
            s = true[e]
            return (f"{ev_text(e)} in slot {s}.",
                    lambda pm, e=e, s=s: pm[e] == s)

        def adjacent(a, b):
            return (f"{ev_text(a)} one slot after {ev_text(b)}.",
                    lambda pm, a=a, b=b: pm[a] == pm[b] + 1)

        def ordered(f, g):
            return (f"{ev_text(f)} before {ev_text(g)}.",
                    lambda pm, f=f, g=g: pm[f] < pm[g])

        stmts = []
        for e in range(K):
            r = rng.random()
            adj = [(a, b) for a in range(K) for b in range(K)
                   if a != b and true[a] == true[b] + 1 and e in (a, b)]
            if r < 0.35 or (r < 0.65 and not adj):
                stmts.append(timed(e))
            elif r < 0.65:
                stmts.append(adjacent(*rng.choice(adj)))
            else:
                j = rng.choice([x for x in range(K) if x != e])
                f, g = (e, j) if true[e] < true[j] else (j, e)
                stmts.append(ordered(f, g))
        rng.shuffle(stmts)

        cons = list(PERMS[K])
        seen = set()
        lines = []
        n = 0
        asked = 0
        wanted = rng.randint(1, 5)
        for text, pred in stmts:
            cons = [pm for pm in cons if pred(pm)]
            n += 1
            lines.append(f"{n} {text}")
            if asked < wanted and n >= 2 and rng.random() < 0.45:
                q = try_question(rng, events, true, cons, K, seen)
                if q is not None:
                    n += 1
                    lines.append(f"{n} {q}")
                    asked += 1
        while asked < wanted:
            q = try_question(rng, events, true, cons, K, seen)
            if q is None:
                break
            n += 1
            lines.append(f"{n} {q}")
            asked += 1
        if asked >= 1:
            return lines


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python gen.py N SEED\n")
        sys.exit(1)
    count = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]))
    out = []
    for _ in range(count):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
