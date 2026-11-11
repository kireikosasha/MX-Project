#!/usr/bin/env python3
"""Synthetic pairing task given as records.

Each record is a row like "qem p1 p2." (pair two persons) or "vaz p3."
(break the pair of one person). Pairs are exclusive: pairing a person who
already has a partner breaks the old pair, so the partner of a person can
change through a chain of earlier pairings and breakings. Queries are
answered by simulating the records above them:
    ozm p1 p2?   yes/no  - are p1 and p2 paired right now
    pmt p1?      p<k>    - the current partner of p1 (only if paired)
    cnt p1?      number  - how many different persons p1 has ever been paired with
"""
import random
import sys

PEOPLE = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
RECORD_OPS = ["qem", "vaz"]
QUERY_OPS = ["ozm", "pmt", "cnt"]


class State:
    def __init__(self):
        self.partner = {}   # person -> current partner
        self.history = {}   # person -> set of everyone ever paired with
        self.mentions = {}  # person -> number of earlier records naming them

    def _unpair(self, x):
        if x in self.partner:
            y = self.partner.pop(x)
            self.partner.pop(y, None)

    def pair(self, a, b):
        self._unpair(a)
        self._unpair(b)
        self.partner[a] = b
        self.partner[b] = a
        self.history.setdefault(a, set()).add(b)
        self.history.setdefault(b, set()).add(a)

    def split(self, a):
        self._unpair(a)

    def note(self, *names):
        for n in names:
            self.mentions[n] = self.mentions.get(n, 0) + 1


def make_record(rng, st, pool):
    op = rng.choice(RECORD_OPS)
    if op == "qem":
        a, b = rng.sample(pool, 2)
        st.pair(a, b)
        st.note(a, b)
        return f"{op} {a} {b}."
    a = rng.choice(pool)
    st.split(a)
    st.note(a)
    return f"{op} {a}."


def make_query(rng, st, pool):
    """Return (text, answer) for a query whose answer is known now, or None."""
    for _ in range(30):
        kind = rng.choice(QUERY_OPS)
        if kind == "ozm":
            a, b = rng.sample(pool, 2)
            if st.mentions.get(a, 0) < 2 or st.mentions.get(b, 0) < 2:
                continue
            ans = "yes" if st.partner.get(a) == b else "no"
            return f"ozm {a} {b}?", ans
        a = rng.choice(pool)
        if st.mentions.get(a, 0) < 2:
            continue
        if kind == "pmt":
            if a not in st.partner:
                continue
            return f"pmt {a}?", st.partner[a]
        # cnt
        if a not in st.history:
            continue
        return f"cnt {a}?", str(len(st.history[a]))
    return None


def make_episode(rng, size):
    """Return a list of (text, answer-or-None) rows, one per row of the episode."""
    while True:
        pool = rng.sample(PEOPLE, rng.randint(3, 8))
        n_rec = rng.randint(3, 12) * size
        n_q = rng.randint(1, 5)
        # query slots: a query is placed after record number k (k >= 2)
        slots = sorted(rng.randint(2, n_rec) for _ in range(n_q))
        st = State()
        rows = []
        qi = 0
        ok = True
        for i in range(1, n_rec + 1):
            rows.append((make_record(rng, st, pool), None))
            while qi < n_q and slots[qi] == i:
                q = make_query(rng, st, pool)
                if q is None:
                    ok = False
                    break
                rows.append(q)
                qi += 1
            if not ok:
                break
        if ok and qi == n_q:
            return rows


def render(rows):
    lines = []
    for num, (text, ans) in enumerate(rows, 1):
        if ans is None:
            lines.append(f"{num} {text}")
        else:
            lines.append(f"{num} {text}\t{ans}")
    return "\n".join(lines)


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < n:
        text = render(make_episode(rng, size))
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
