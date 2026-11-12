#!/usr/bin/env python3
"""Synthetic record task: weighted ballots are cast, retracted and re-weighted;
queries ask a total, a comparison of two options, or the voters behind an option."""
import itertools
import random
import sys

VOTERS = ["v1", "v2", "v3", "v4", "v5", "v6"]
OPTIONS = ["o1", "o2", "o3", "o4"]


class State:
    def __init__(self):
        self.ballot = {}   # voter -> option currently chosen
        self.weight = {}   # voter -> explicitly set weight (default 1)

    def w(self, v):
        return self.weight.get(v, 1)

    def total(self, o):
        return sum(self.w(v) for v, b in self.ballot.items() if b == o)

    def support(self, voters):
        # records that determine these voters' contributions: the cast and,
        # if present, the weight record of each voter
        return sum(1 + (1 if v in self.weight else 0) for v in voters)


def make_record(rng, st, size):
    r = rng.random()
    if r < 0.5:
        v = rng.choice(VOTERS)
        o = rng.choice(OPTIONS)
        st.ballot[v] = o
        return f"cst {v} {o}."
    if r < 0.7:
        v = rng.choice(VOTERS)
        st.ballot.pop(v, None)
        return f"rtr {v}."
    v = rng.choice(VOTERS)
    n = rng.randint(1, 5 * size)
    st.weight[v] = n
    return f"wgt {v} {n}."


def query_candidates(st):
    out = []
    for o in OPTIONS:
        voters = sorted(v for v, b in st.ballot.items() if b == o)
        if voters and st.support(voters) >= 2:
            out.append((f"qtl {o}?", str(st.total(o))))
        if len(voters) >= 2:
            out.append((f"qwh {o}?", ",".join(voters)))
    for a, b in itertools.combinations(OPTIONS, 2):
        ta, tb = st.total(a), st.total(b)
        if ta == tb:
            continue
        winner = a if ta > tb else b
        voters = [v for v, bl in st.ballot.items() if bl in (a, b)]
        if st.support(voters) >= 2:
            out.append((f"qld {a} {b}?", winner))
    return out


def make_episode(rng, size):
    for _ in range(100000):
        st = State()
        n_rec = rng.randint(3, 12) * size
        k = rng.randint(1, 5)
        total = n_rec + k
        qpos = set(rng.sample(range(2, total), k))
        rows = []
        used = set()
        failed = False
        for i in range(total):
            if i in qpos:
                cands = [(t, a) for (t, a) in query_candidates(st) if t not in used]
                if not cands:
                    failed = True
                    break
                t, a = rng.choice(cands)
                used.add(t)
                rows.append(f"{i + 1} {t}\t{a}")
            else:
                rows.append(f"{i + 1} {make_record(rng, st, size)}")
        if not failed:
            return rows
    raise RuntimeError("could not generate an episode")


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    seen = set()
    episodes = []
    while len(episodes) < n:
        ep = make_episode(rng, size)
        key = "\n".join(ep)
        if key in seen:
            continue
        seen.add(key)
        episodes.append(ep)
    for ep in episodes:
        for line in ep:
            print(line)


if __name__ == "__main__":
    main(sys.argv)
