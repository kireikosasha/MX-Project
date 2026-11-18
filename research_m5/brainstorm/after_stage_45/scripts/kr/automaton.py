#!/usr/bin/env python3
"""
One learner for counters, resets and composition tables: a soft finite automaton whose turn is a row-stochastic
matrix per input symbol (mass conserved, as in loop.py), taught by answers on short sequences (length 1..10),
then made hard (argmax per row) and run exactly on sequences of length 100 and 500.
The number of states is not told: the smallest one whose hard automaton explains all teaching sequences is kept.

Tasks (regular ones from Deletang et al. 2023 and Liu et al. 2023, plus the analogues of the three machines of one.py):
  parity      group Z2           (counter)
  mod5        group Z5           (counter: sum of digits mod 5)
  cycle       group Z5           (moves -1/0/+1 on a cycle of 5)
  latest      reset / flip-flop  (write 0/1/2 or do nothing; say the latest written: "where is X now")
  s3          non-abelian group  (product of permutations of three: a composition table, like kin relations)
  firstlast   product of a reset and a reset (first symbol == last symbol: Deletang's even pairs)
  majority    NOT regular        (sign of #a - #b: needs an unbounded counter, the register of walk.py)
"""
import itertools
import random
import sys
import time

import torch

torch.set_num_threads(1)
PERMS = list(itertools.permutations(range(3)))


def compose(p, q):
    return tuple(q[p[i]] for i in range(3))


def label(task, xs):
    if task == "parity":
        return sum(xs) % 2
    if task == "mod5":
        return sum(xs) % 5
    if task == "cycle":
        return sum(x - 1 for x in xs) % 5
    if task == "latest":
        v = 0
        for x in xs:
            v = x if x < 3 else v
        return v
    if task == "s3":
        p = PERMS[0]
        for x in xs:
            p = compose(p, PERMS[x])
        return PERMS.index(p)
    if task == "firstlast":
        return int(xs[0] == xs[-1])
    if task == "majority":
        d = sum(1 if x == 0 else -1 for x in xs)
        return (d > 0) - (d < 0) + 1


SIGMA = {"parity": 2, "mod5": 5, "cycle": 3, "latest": 4, "s3": 6, "firstlast": 2, "majority": 2}
OUT = {"parity": 2, "mod5": 5, "cycle": 5, "latest": 3, "s3": 6, "firstlast": 2, "majority": 3}


def sample(task, n, lo, hi, rng):
    data = []
    for _ in range(n):
        L = rng.randint(lo, hi)
        xs = [rng.randrange(SIGMA[task]) for _ in range(L)]
        data.append((xs, label(task, xs)))
    return data


def batch(data):
    L = max(len(x) for x, _ in data)
    xs = torch.full((len(data), L), -1, dtype=torch.long)
    for i, (x, _) in enumerate(data):
        xs[i, :len(x)] = torch.tensor(x)
    return xs, torch.tensor([y for _, y in data])


class Soft(torch.nn.Module):
    def __init__(self, S, sigma, out, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T = torch.nn.Parameter(torch.randn(sigma, S, S, generator=g))
        self.O = torch.nn.Parameter(0.1 * torch.randn(S, out, generator=g))
        self.S = S

    def forward(self, xs):
        b, L = xs.shape
        T = torch.softmax(self.T, -1)                           # every row a distribution: mass is carried, none made or lost
        p = torch.zeros(b, self.S)
        p[:, 0] = 1
        for t in range(L):
            on = xs[:, t] >= 0
            allsteps = torch.einsum('bs,ast->bat', p, T)
            step = allsteps[torch.arange(b), xs[:, t].clamp(min=0)]
            p = torch.where(on[:, None], step, p)
        return p @ torch.softmax(self.O, -1), T

    def hard(self):
        return self.T.argmax(-1).tolist(), self.O.argmax(-1).tolist()


def run_hard(h, xs):
    T, O = h
    s = 0
    for x in xs:
        s = T[x][s]
    return O[s]


def polish(h, data, rounds=3):
    """every hard choice tried over, exactly, on the teaching sequences (as walk.py polish)"""
    T, O = h
    def score():
        return sum(run_hard((T, O), x) == y for x, y in data)
    best = score()
    for _ in range(rounds):
        changed = False
        for a in range(len(T)):
            for s in range(len(T[a])):
                was = T[a][s]
                for s2 in range(len(T[a])):
                    T[a][s] = s2
                    sc = score()
                    if sc > best:
                        best, was, changed = sc, s2, True
                T[a][s] = was
        # what is said: commonest answer per reached state
        from collections import Counter
        seen = {}
        for x, y in data:
            st = 0
            for c in x:
                st = T[c][st]
            seen.setdefault(st, Counter())[y] += 1
        for st, c in seen.items():
            O[st] = c.most_common(1)[0][0]
        sc = score()
        if sc >= best:
            best = sc
        if not changed:
            break
    return (T, O), best


def learn(task, S, seed, teach, steps=400):
    torch.manual_seed(seed)
    m = Soft(S, SIGMA[task], OUT[task], seed)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    rng = random.Random(seed)
    for step in range(steps):
        xs, ys = batch(rng.sample(teach, 256))
        q, T = m(xs)
        loss = -torch.log(q[torch.arange(len(ys)), ys].clamp_min(1e-9)).mean()
        ent = -(T * torch.log(T.clamp_min(1e-9))).sum(-1).mean()
        loss = loss + 0.05 * (step / steps) * ent              # pushed towards hard choices late in teaching
        opt.zero_grad()
        loss.backward()
        opt.step()
    h, right = polish(m.hard(), teach[:400])
    return h, right / min(400, len(teach))


def main():
    tasks = sys.argv[1:] or list(SIGMA)
    rng = random.Random(0)
    for task in tasks:
        t0 = time.time()
        teach = sample(task, 2000, 1, 10, rng)
        long100 = sample(task, 1000, 100, 100, rng)
        long500 = sample(task, 300, 500, 500, rng)
        kept = None
        for S in (2, 3, 4, 5, 6, 8, 12):
            for seed in range(4):
                h, fit = learn(task, S, seed, teach)
                if fit == 1.0:
                    kept = (S, seed, h)
                    break
            if kept:
                break
        if kept is None:
            S, seed = 12, -1
            print(f"{task:10s} no automaton up to 12 states explains all teaching sequences (best fit {fit:.3f}); "
                  f"on length 100: {100 * sum(run_hard(h, x) == y for x, y in long100) / len(long100):.1f}%  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            continue
        S, seed, h = kept
        a100 = 100 * sum(run_hard(h, x) == y for x, y in long100) / len(long100)
        a500 = 100 * sum(run_hard(h, x) == y for x, y in long500) / len(long500)
        print(f"{task:10s} kept {S} states (beginning {seed}); taught on length 1-10; "
              f"exact on length 100: {a100:.1f}%, on 500: {a500:.1f}%  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
