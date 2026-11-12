#!/usr/bin/env python3
"""Generate synthetic record-log reasoning episodes.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1 (default), 3 or 10)

Each episode is a log of rows. A row is either a record (a state change, ends
with a full stop) or a query (ends with a question mark, then TAB, then the
answer). Some records are rules and some are facts:

  zov g1 g2.   rule:  every member of g1 is a member of g2
  qep g1 g2.   rule removed
  ruk u3 g1.   fact:  u3 is directly in g1
  vem u3 g1.   fact removed
  jaq u3 7.    fact:  u3 has base score 7 (latest value wins)
  pyl g1 5.    rule:  every member of g1 gets 5 bonus points (latest wins)

Queries are answered from the state built by the records above them:

  heq u3 g2?   yes / no   (is u3 a member of g2, directly or through rules)
  lok u3?      g1,g2      (sorted groups u3 belongs to)
  scu u3?      14         (base score plus bonuses of all groups u3 belongs to)

All randomness comes from random.Random(SEED). Standard library only.
"""
import random
import sys
from collections import deque

# Record operation codes (6 in total).
OP_SUB, OP_CUT, OP_IN, OP_OUT, OP_BASE, OP_BONUS = "zov", "qep", "ruk", "vem", "jaq", "pyl"
REC_OPS = [OP_SUB, OP_CUT, OP_IN, OP_OUT, OP_BASE, OP_BONUS]
REC_WEIGHTS = [0.25, 0.07, 0.30, 0.08, 0.15, 0.15]

# Query operation codes (3 in total).
Q_HAS, Q_LIST, Q_SCORE = "heq", "lok", "scu"
Q_OPS = [Q_HAS, Q_LIST, Q_SCORE]
Q_WEIGHTS = [0.4, 0.3, 0.3]

N_IND = 8  # individuals u1..u8
N_GRP = 8  # groups g1..g8


class Retry(Exception):
    """Raised when an episode cannot be completed; the episode is regenerated."""


class World:
    """State built by the records so far. Answers are computed from it."""

    def __init__(self):
        self.rules = set()  # (a, b): every member of group a is a member of group b
        self.facts = set()  # (u, g): individual u is directly in group g
        self.base = {}      # u -> base score
        self.bonus = {}     # g -> bonus per member

    def closure(self, u):
        """Map each group of u to its derivation depth.

        Depth 0 means a direct fact; depth k means reached through k rule
        applications along a shortest route.
        """
        depth = {}
        q = deque()
        for g in sorted(g for (x, g) in self.facts if x == u):
            depth[g] = 0
            q.append(g)
        rules = sorted(self.rules)
        while q:
            g = q.popleft()
            for a, b in rules:
                if a == g and b not in depth:
                    depth[b] = depth[g] + 1
                    q.append(b)
        return depth

    def score(self, u):
        depth = self.closure(u)
        return self.base.get(u, 0) + sum(self.bonus[g] for g in depth if g in self.bonus)


def make_record(rng, world, num_max):
    for _ in range(200):
        op = rng.choices(REC_OPS, REC_WEIGHTS)[0]
        if op == OP_SUB:
            a, b = rng.sample(range(1, N_GRP + 1), 2)
            if (a, b) in world.rules:
                continue
            world.rules.add((a, b))
            return f"{op} g{a} g{b}."
        if op == OP_CUT:
            if not world.rules:
                continue
            a, b = rng.choice(sorted(world.rules))
            world.rules.remove((a, b))
            return f"{op} g{a} g{b}."
        if op == OP_IN:
            u, g = rng.randint(1, N_IND), rng.randint(1, N_GRP)
            if (u, g) in world.facts:
                continue
            world.facts.add((u, g))
            return f"{op} u{u} g{g}."
        if op == OP_OUT:
            if not world.facts:
                continue
            u, g = rng.choice(sorted(world.facts))
            world.facts.remove((u, g))
            return f"{op} u{u} g{g}."
        if op == OP_BASE:
            u, n = rng.randint(1, N_IND), rng.randint(1, num_max)
            world.base[u] = n
            return f"{op} u{u} {n}."
        if op == OP_BONUS:
            g, n = rng.randint(1, N_GRP), rng.randint(1, num_max)
            world.bonus[g] = n
            return f"{op} g{g} {n}."
    raise Retry()


def make_query(rng, world):
    """Return (question_text, answer) for the current state, or None."""
    members = sorted({u for (u, _) in world.facts})
    if not members:
        return None
    for _ in range(100):
        op = rng.choices(Q_OPS, Q_WEIGHTS)[0]
        if op == Q_HAS:
            u = rng.choice(members)
            depth = world.closure(u)
            derived = sorted(g for g, d in depth.items() if d >= 1)
            if derived and rng.random() < 0.7:
                g = rng.choice(derived)
            else:
                g = rng.randint(1, N_GRP)
            ans = "yes" if g in depth else "no"
            return f"{op} u{u} g{g}?", ans
        if op == Q_LIST:
            cands = [u for u in members if any(d >= 1 for d in world.closure(u).values())]
            if not cands:
                continue
            u = rng.choice(cands)
            depth = world.closure(u)
            return f"{op} u{u}?", ",".join(f"g{g}" for g in sorted(depth))
        if op == Q_SCORE:
            cands = [u for u in members if any(g in world.bonus for g in world.closure(u))]
            if not cands:
                continue
            u = rng.choice(cands)
            return f"{op} u{u}?", str(world.score(u))
    return None


def make_episode(rng, size):
    world = World()
    num_max = 9 * size
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    total = n_rec + n_q
    q_slots = set(rng.sample(range(2, total), n_q))
    lines = []
    for i in range(total):
        num = i + 1
        if i in q_slots:
            made = make_query(rng, world)
            if made is None:
                raise Retry()
            text, ans = made
            lines.append(f"{num} {text}\t{ans}")
        else:
            lines.append(f"{num} {make_record(rng, world, num_max)}")
    return lines


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        for _attempt in range(10000):
            try:
                lines = make_episode(rng, size)
            except Retry:
                continue
            key = "\n".join(lines)
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError("could not generate a distinct episode")
        out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
