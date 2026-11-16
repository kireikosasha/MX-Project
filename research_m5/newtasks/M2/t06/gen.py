"""Synthetic order-in-time reading task.

Usage: python gen.py N SEED   (prints N stories to standard output)

Each story is a stream of movement events, told out of order. Each event is
"NAME went to the PLACE", optionally tied to one earlier event with
"after" or "before". The true times come from a hidden simulation; the
questions are answered from that simulation, and only kept when the
statements shown so far already fix the answer.
"""
import random
import sys

NAMES = ["Ann", "Bob", "Cat", "Dan", "Eve"]
PLACES = ["kitchen", "garden", "hall", "office", "shed"]
NOWHERE = "nowhere"


def closure(n, pairs):
    """lt[i][j] is True when event i is known to be before event j."""
    lt = [[False] * n for _ in range(n)]
    for i, j in pairs:
        lt[i][j] = True
    for m in range(n):
        for i in range(n):
            if lt[i][m]:
                for j in range(n):
                    if lt[m][j]:
                        lt[i][j] = True
    return lt


class World:
    def __init__(self, rng):
        self.n = rng.randint(3, 12)
        combos = [(p, q) for p in NAMES for q in PLACES]
        # event i is "ev[i][0] went to the ev[i][1]"; line order is event index order
        self.ev = rng.sample(combos, self.n)
        self.time = list(range(self.n))  # hidden true time of each event
        rng.shuffle(self.time)
        self.stmts = []
        self.rels = []  # (statement index, a, b): event a is before event b
        for i, (person, place) in enumerate(self.ev):
            if i > 0 and rng.random() < 0.7:
                f = rng.randrange(i)
                fp, fq = self.ev[f]
                if self.time[i] > self.time[f]:
                    text = f"{person} went to the {place} after {fp} went to the {fq}."
                    self.rels.append((i, f, i))
                else:
                    text = f"{person} went to the {place} before {fp} went to the {fq}."
                    self.rels.append((i, i, f))
            else:
                text = f"{person} went to the {place}."
            self.stmts.append(text)

    def order(self, k):
        """Known order among the first k events, from the statements above."""
        return closure(k, [(a, b) for (s, a, b) in self.rels if s < k])

    # Answers that the statements fix (None when they do not).
    def known_now(self, p, k, lt):
        evs = [u for u in range(k) if self.ev[u][0] == p]
        if not evs:
            return NOWHERE
        for m in evs:
            if all(u == m or lt[u][m] for u in evs):
                return self.ev[m][1]
        return None

    def known_at(self, p, e, k, lt):
        cand = [u for u in range(k) if self.ev[u][0] == p and u != e]
        possible = [u for u in cand if not lt[e][u]]  # not known to be after e
        if not possible:
            return NOWHERE
        for m in possible:
            if lt[m][e] and all(u == m or lt[u][m] for u in possible):
                return self.ev[m][1]
        return None

    # Answers from the hidden simulation.
    def true_now(self, p, k):
        evs = [u for u in range(k) if self.ev[u][0] == p]
        if not evs:
            return NOWHERE
        m = max(evs, key=lambda u: self.time[u])
        return self.ev[m][1]

    def true_at(self, p, e, k):
        cand = [u for u in range(k)
                if self.ev[u][0] == p and u != e and self.time[u] < self.time[e]]
        if not cand:
            return NOWHERE
        m = max(cand, key=lambda u: self.time[u])
        return self.ev[m][1]


def who_answer(locs, place):
    names = sorted(p.lower() for p, loc in zip(NAMES, locs) if loc == place)
    return ",".join(names) if names else "nobody"


def question(w, rng, k, lt):
    """Try one random question about the first k events. Returns (text, answer) or None."""
    kind = rng.randrange(4)
    if kind == 0:
        p = rng.choice(NAMES)
        text = f"Where is {p} now?"
        det = w.known_now(p, k, lt)
        true = w.true_now(p, k)
    elif kind == 1:
        e = rng.randrange(k)
        m, pl = w.ev[e]
        p = rng.choice([x for x in NAMES if x != m])
        text = f"Where was {p} when {m} went to the {pl}?"
        det = w.known_at(p, e, k, lt)
        true = w.true_at(p, e, k)
    elif kind == 2:
        e = rng.randrange(k)
        p, q = w.ev[e]
        pl = rng.choice(PLACES)
        text = f"Who was in the {pl} before {p} went to the {q}?"
        dets = [w.known_at(x, e, k, lt) for x in NAMES]
        trues = [w.true_at(x, e, k) for x in NAMES]
        det = None if None in dets else who_answer(dets, pl)
        true = who_answer(trues, pl)
    else:
        if k < 2:
            return None
        a, b = rng.sample(range(k), 2)
        p, pl = w.ev[a]
        m, q = w.ev[b]
        text = f"Did {p} go to the {pl} before {m} went to the {q}?"
        if lt[a][b]:
            det = "yes"
        elif lt[b][a]:
            det = "no"
        else:
            det = None
        true = "yes" if w.time[a] < w.time[b] else "no"
    if det is None:
        return None
    if det != true:
        raise RuntimeError("statements do not fix the answer the simulation gives")
    return text, true


def make_story(rng):
    while True:
        w = World(rng)
        qpos = [rng.randint(1, w.n) for _ in range(rng.randint(1, 5))]
        lines = []  # (text, answer or None)
        for k in range(1, w.n + 1):
            lines.append((w.stmts[k - 1], None))
            for _ in range(qpos.count(k)):
                lt = w.order(k)
                for _ in range(30):
                    res = question(w, rng, k, lt)
                    if res:
                        lines.append(res)
                        break
        if any(ans is not None for _, ans in lines):
            return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(count):
        for i, (text, ans) in enumerate(make_story(rng), 1):
            if ans is None:
                print(f"{i} {text}")
            else:
                print(f"{i} {text}\t{ans}")


if __name__ == "__main__":
    main()
