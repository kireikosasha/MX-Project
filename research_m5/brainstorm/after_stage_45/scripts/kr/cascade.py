#!/usr/bin/env python3
"""
The same learner with one more component, in cascade: an integer counter whose move at every symbol (-1, 0, +1, back
to nought) is chosen by the symbol and the state the finite part has just come to. What is said: a table from (finite
state, sign of the counter), or the counter itself - a learnt choice. Soft over (state, counter value) with all mass
carried; then hard, and run on whole numbers on sequences of length 100 and 500.

Kept: the smallest finite part that explains all teaching sequences; the counter is used only if without it nothing
explains them (a counter costs more than any finite part up to 12 states).

  majority     sign of #a - #b                       (not regular: the register of walk.py)
  since        how many a since the last c           (a counter with a reset: "how many does X carry now", bAbI 7)
  s3, latest   regular: the counter must stay unused
"""
import random
import sys
import time

import torch

import automaton as A

torch.set_num_threads(1)
M = 12                                                          # teaching lengths up to 10: the counter lives in [-M, M]


def label(task, xs):
    if task == "since":
        n = 0
        for x in xs:
            n = 0 if x == 2 else n + (x == 0)
        return n
    return A.label(task, xs)


SIGMA = dict(A.SIGMA, since=3)
OUT = dict(A.OUT, since=M + 1)


def sample(task, n, lo, hi, rng):
    out = []
    for _ in range(n):
        xs = [rng.randrange(SIGMA[task]) for _ in range(rng.randint(lo, hi))]
        out.append((xs, label(task, xs)))
    return out


class Soft(torch.nn.Module):
    def __init__(self, S, sigma, out, seed, counter):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T = torch.nn.Parameter(torch.randn(sigma, S, S, generator=g))
        self.U = torch.nn.Parameter(torch.randn(sigma, S, 4, generator=g))        # -1, 0, +1, back to nought
        self.O = torch.nn.Parameter(0.1 * torch.randn(S, 3, out, generator=g))     # (state, sign) -> answer
        self.R = torch.nn.Parameter(torch.zeros(2))                                 # say the table / say the counter
        self.S, self.counter, self.out = S, counter, out

    def forward(self, xs):
        b, L = xs.shape
        T = torch.softmax(self.T, -1)
        U = torch.softmax(self.U, -1)
        C = 2 * M + 1
        p = torch.zeros(b, self.S, C)
        p[:, 0, M] = 1
        for t in range(L):
            on = xs[:, t] >= 0
            x = xs[:, t].clamp(min=0)
            q = torch.einsum('bsc,bst->btc', p, T[x])                 # the finite part moves
            if self.counter:
                u = U[x]                                              # [b, S, 4]: the counter's move, by the new state
                down = torch.cat([q[:, :, 1:], torch.zeros(b, self.S, 1)], 2)
                up = torch.cat([torch.zeros(b, self.S, 1), q[:, :, :-1]], 2)
                zero = torch.zeros_like(q)
                zero[:, :, M] = q.sum(2)
                q = u[:, :, 0, None] * down + u[:, :, 1, None] * q + u[:, :, 2, None] * up + u[:, :, 3, None] * zero
            p = torch.where(on[:, None, None], q, p)
        sign = torch.stack([p[:, :, :M].sum(2), p[:, :, M], p[:, :, M + 1:].sum(2)], 2)   # [b, S, 3]
        table = torch.einsum('bsg,sga->ba', sign, torch.softmax(self.O, -1))
        r = torch.softmax(self.R, 0) if self.counter else torch.tensor([1.0, 0.0])
        value = torch.zeros(b, self.out)
        n = min(self.out, M + 1)
        value[:, :n] = p.sum(1)[:, M:M + n]
        return r[0] * table + r[1] * value, T

    def hard(self):
        return (self.T.argmax(-1).tolist(), self.U.argmax(-1).tolist() if self.counter else None,
                self.O.argmax(-1).tolist(), int(self.R.argmax()) if self.counter else 0)


def run(h, xs):
    T, U, O, r = h
    s, c = 0, 0
    for x in xs:
        s = T[x][s]
        if U is not None:
            c = (c - 1, c, c + 1, 0)[U[x][s]]
    return c if r == 1 else O[s][(c > 0) - (c < 0) + 1]


def learn(task, S, seed, teach, counter, steps=400):
    torch.manual_seed(seed)
    m = Soft(S, SIGMA[task], OUT[task], seed, counter)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    rng = random.Random(seed)
    for step in range(steps):
        xs, ys = A.batch(rng.sample(teach, 256))
        q, T = m(xs)
        loss = -torch.log(q[torch.arange(len(ys)), ys].clamp_min(1e-9)).mean()
        loss = loss + 0.05 * (step / steps) * -(T * torch.log(T.clamp_min(1e-9))).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    h = m.hard()
    return h, sum(run(h, x) == y for x, y in teach[:400]) / 400


def main():
    rng = random.Random(0)
    for task in sys.argv[1:]:
        t0 = time.time()
        teach = sample(task, 2000, 1, 10, rng)
        long100, long500 = sample(task, 1000, 100, 100, rng), sample(task, 300, 500, 500, rng)
        kept, best = None, 0
        for counter in (False, True):
            for S in (1, 2, 3, 4, 6, 8, 12):
                if counter and S > 4:
                    break
                for seed in range(4):
                    h, fit = learn(task, S, seed, teach, counter)
                    best = max(best, fit)
                    if fit == 1.0:
                        kept = (counter, S, seed, h)
                        break
                if kept:
                    break
            if kept:
                break
        if not kept:
            print(f"{task:9s} nothing explains all teaching sequences (best {best:.3f})", flush=True)
            continue
        counter, S, seed, h = kept
        a100 = 100 * sum(run(h, x) == y for x, y in long100) / len(long100)
        a500 = 100 * sum(run(h, x) == y for x, y in long500) / len(long500)
        print(f"{task:9s} kept {S} finite states {'+ a counter' if counter else '(no counter)'}"
              f"{', says the counter' if counter and h[3] == 1 else ''}; taught on 1-10; exact on 100: {a100:.1f}%, on 500: {a500:.1f}%  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
