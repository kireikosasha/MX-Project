"""Synthetic order-in-time reading task generator.

Stories tell visits ("Ann went to the park") out of order: some with a clock
time, some only relative to another visit ("after" / "before"). Questions ask
where someone was at a time or when another visit happened, who was at a place
at a time, who came to a place first, or whether one visit came before another.

Answers come from a hidden simulation (every visit has a hidden clock time).
A question is only asked when the statements above it fix its answer (checked
over all orders of the revealed visits that respect the statements) and that
answer equals the hidden truth.

Usage: python gen.py N SEED
"""
import random
import sys

NAMES = ["ann", "ben", "cara", "dan", "eve", "finn"]
PLACES = ["park", "school", "shop", "lake", "farm"]
HOME = "home"
MAX_TIME = 30
QK = ("?", "?")  # pseudo visit used for a question moment


def cap(word):
    return word[0].upper() + word[1:]


def make_truth(rng):
    people = rng.sample(NAMES, rng.randint(2, 4))
    pairs = [(p, pl) for p in people for pl in PLACES]
    visits = rng.sample(pairs, rng.randint(3, 6))
    times = rng.sample(range(1, MAX_TIME + 1), len(visits))
    hidden = {(p, HOME): 0 for p in people}  # start at home, time 0
    for key, tm in zip(visits, times):
        hidden[key] = tm
    return people, visits, hidden


def make_statements(rng, people, visits, hidden):
    stmts = [("start", (p, HOME)) for p in people]
    for key in visits:
        if rng.random() < 0.4:
            other = rng.choice([v for v in visits if v != key])
            rel = "after" if hidden[other] < hidden[key] else "before"
            stmts.append(("rel", key, other, rel))
        else:
            stmts.append(("timed", key))
    for _ in range(rng.randint(0, 2)):
        a, b = rng.sample(visits, 2)
        rel = "after" if hidden[b] < hidden[a] else "before"
        stmts.append(("rel", a, b, rel))
    rng.shuffle(stmts)
    return stmts


def render(rng, s, hidden):
    if s[0] == "start":
        return f"{cap(s[1][0])} started at the home."
    if s[0] == "timed":
        p, pl = s[1]
        tm = hidden[s[1]]
        if rng.random() < 0.5:
            return f"{cap(p)} went to the {pl} at time {tm}."
        return f"At time {tm} {cap(p)} went to the {pl}."
    _, (p, pl), (q, r), rel = s
    return f"{cap(p)} went to the {pl} {rel} {cap(q)} went to the {r}."


def reveal(state, s, hidden):
    keys, timed, rels = state

    def add(k):
        if k not in keys:
            keys.append(k)

    if s[0] == "start":
        add(s[1])
        timed[s[1]] = 0
    elif s[0] == "timed":
        add(s[1])
        timed[s[1]] = hidden[s[1]]
    else:
        _, key, other, rel = s
        add(key)
        add(other)
        if rel == "after":
            rels.append((other, key))
        else:
            rels.append((key, other))


def closure(keys, timed, rels):
    """Transitive closure of the 'happened before' relation as bitmasks."""
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    reach = [0] * n

    def edge(a, b):
        reach[idx[a]] |= 1 << idx[b]

    for a in keys:
        if a in timed:
            for b in keys:
                if b in timed and timed[a] < timed[b]:
                    edge(a, b)
    for a, b in rels:
        edge(a, b)
    for k in keys:  # a person's start comes before that person's visits
        start = (k[0], HOME)
        if k[1] != HOME and start in idx:
            edge(start, k)
    for m in range(n):
        bit = 1 << m
        for i in range(n):
            if reach[i] & bit:
                reach[i] |= reach[m]
    return idx, reach


def before_(idx, reach, a, b):
    return ((reach[idx[a]] >> idx[b]) & 1) == 1


def rev_loc(p, qk, keys, idx, reach):
    """Place of p just before moment qk, if the revealed facts fix it."""
    if qk not in idx:
        return None
    before = []
    for k in keys:
        if k[0] != p or k == qk:
            continue
        if before_(idx, reach, k, qk):
            before.append(k)
        elif not before_(idx, reach, qk, k):
            return None
    if not before:
        return None
    tops = [b for b in before
            if all(c == b or before_(idx, reach, c, b) for c in before)]
    if len(tops) != 1:
        return None
    return tops[0][1]


def truth_loc(p, tm, visits, hidden):
    best, where = -1, HOME
    for key in visits:
        if key[0] == p and best < hidden[key] < tm:
            best, where = hidden[key], key[1]
    return where


def try_question(rng, people, visits, hidden, state):
    keys, timed, rels = state
    used = {hidden[v] for v in visits}
    free = [x for x in range(1, MAX_TIME + 1) if x not in used]
    shown = [k for k in keys if k[1] != HOME]
    kind = rng.randrange(5)
    if kind in (0, 4):
        tm = rng.choice(free)
        ks = keys + [QK]
        tt = dict(timed)
        tt[QK] = tm
        idx, reach = closure(ks, tt, rels)
    else:
        ks = keys
        idx, reach = closure(keys, timed, rels)

    if kind == 0:  # Where was X at time t?
        p = rng.choice(people)
        got = rev_loc(p, QK, ks, idx, reach)
        want = truth_loc(p, tm, visits, hidden)
        if got is None or got != want:
            return None
        return f"Where was {cap(p)} at time {tm}?", want

    if kind == 1:  # Where was X when Y went to the Z?
        if not shown:
            return None
        e = rng.choice(shown)
        p = rng.choice([x for x in people if x != e[0]])
        got = rev_loc(p, e, ks, idx, reach)
        want = truth_loc(p, hidden[e], visits, hidden)
        if got is None or got != want:
            return None
        return f"Where was {cap(p)} when {cap(e[0])} went to the {e[1]}?", want

    if kind == 2:  # Did X go to the P before/after Y went to the Q?
        if len(shown) < 2:
            return None
        a, b = rng.sample(shown, 2)
        ab = before_(idx, reach, a, b)
        if ab == before_(idx, reach, b, a):
            return None
        mode = rng.choice(["before", "after"])

        def ans(a_first):
            return "yes" if a_first == (mode == "before") else "no"

        got = ans(ab)
        want = ans(hidden[a] < hidden[b])
        if got != want:
            return None
        text = (f"Did {cap(a[0])} go to the {a[1]} {mode} "
                f"{cap(b[0])} went to the {b[1]}?")
        return text, want

    if kind == 3:  # Who went to the P first?
        pl = rng.choice(PLACES)
        group = [k for k in shown if k[1] == pl]
        if not group:
            return None
        tops = [s for s in group
                if all(c == s or before_(idx, reach, s, c) for c in group)]
        if len(tops) != 1:
            return None
        got = tops[0][0]
        want = min([v for v in visits if v[1] == pl],
                   key=lambda v: hidden[v])[0]
        if got != want:
            return None
        return f"Who went to the {pl} first?", want

    # kind 4: Who was at the P at time t?
    pl = rng.choice(PLACES)
    got_names = []
    for p in people:
        loc = rev_loc(p, QK, ks, idx, reach)
        if loc is None:
            return None
        if loc == pl:
            got_names.append(p)
    want_names = [p for p in people if truth_loc(p, tm, visits, hidden) == pl]
    got = ",".join(sorted(got_names)) or "nobody"
    want = ",".join(sorted(want_names)) or "nobody"
    if got != want:
        return None
    return f"Who was at the {pl} at time {tm}?", want


def ask(rng, people, visits, hidden, state):
    for _ in range(3):
        q = try_question(rng, people, visits, hidden, state)
        if q:
            return q
    return None


def make_story(rng):
    while True:
        people, visits, hidden = make_truth(rng)
        stmts = make_statements(rng, people, visits, hidden)
        state = ([], {}, [])
        lines = []
        nq = rng.randint(1, 5)
        asked = 0
        for s in stmts:
            lines.append(f"{len(lines) + 1} {render(rng, s, hidden)}")
            reveal(state, s, hidden)
            if asked < nq and rng.random() < 0.5:
                q = ask(rng, people, visits, hidden, state)
                if q:
                    lines.append(f"{len(lines) + 1} {q[0]}\t{q[1]}")
                    asked += 1
        tries = 0
        while asked < nq and tries < 6:
            tries += 1
            q = ask(rng, people, visits, hidden, state)
            if q:
                lines.append(f"{len(lines) + 1} {q[0]}\t{q[1]}")
                asked += 1
        if asked:
            return lines


def main():
    n_stories = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n_stories):
        out.extend(make_story(rng))
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    main()
