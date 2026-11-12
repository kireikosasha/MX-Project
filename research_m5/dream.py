#!/usr/bin/env python3
"""
Tasks the ground makes up for itself, to teach the hint network (prior.py) moves it has seldom met.

The network learns which reading of a row to try first from tasks already solved; of rare moves (a link with a
direction, the smaller of two amounts, a union of what two things lead to) there are few among those, and for rows
that need them it has no hint. But the ground knows all its moves. Here it plays: takes some kinds of things, makes
up a few kinds of records, gives each a move at random - every move as often as any other - makes up queries that
read what these moves write, plays episodes and writes down the answers. The result is a task with a known reading,
as many as wanted, with no author. The network is taught on such tasks (alone or together with the solved ones).

    python dream.py --tasks 400 --out dreams.json                 # the made-up rows with their readings
    python dream.py --tasks 400 --real new/N:results/N.txt,...    # how a network taught on dreams places real readings
"""
import argparse
import json
import os
import random

import numpy as np

import prior
import records

WRITES = {"F": ("set", "pair", "setat", "claim", "wait", "copy"), "E": ("tie", "tieboth", "tie2"), "Q": ("qset", "qadd", "qpass"), "L": ("link",)}
THEMES = ("F", "F", "F", "E", "E", "Q", "Q", "FQ", "FQ", "FE", "L", "EQ")
LETTERS = "abcdefghjkmnpqrstuvwxyz"


def code(rng, taken):
    while True:
        op = "".join(rng.choice("bdfgjklmnprstvxz") + rng.choice("aeiou") for _ in range(2))[:rng.choice((2, 3, 4))]
        if op not in taken:
            taken.add(op)
            return op


def make(rng):
    """a made-up task -> (the reading of every kind of record, the program of every kind of query, the names of things)"""
    L = records.Learner([])
    kinds = rng.sample(LETTERS, rng.randint(1, 3))
    names = {k: [f"{k}{i}" for i in range(1, rng.choice((4, 6, 8)) + 1)] for k in kinds}
    theme = rng.choice(THEMES)
    taken, rec = set(), {}
    for _ in range(rng.randint(2, 5)):
        nt, nn = rng.choice((1, 2, 2, 2, 3)), rng.choice((0, 0, 0, 1, 1, 2))
        form = (code(rng, taken), False, tuple(rng.choice(kinds) for _ in range(nt)) + ("#",) * nn)
        rows = [p for p in L.rows(nt, nn, {s: 0 for s in records.K}, theme + "T") if all(m[1] == 0 for m in p)]
        if not rows:
            continue
        by = {}
        for p in rows:                                           # every move as often as any other
            by.setdefault(p[0][0], []).append(p)
        rec[form] = rng.choice(by[rng.choice(sorted(by))])
    if not any(m[0] in WRITES[s] for p in rec.values() for m in p for s in theme):
        return None
    L.names = names
    types = L.types(rec)
    ask = {}
    for _ in range(rng.randint(1, 3)):
        nt, nn = rng.choice((1, 1, 2, 2, 3)), rng.choice((0, 0, 0, 1))
        form = (code(rng, taken), True, tuple(rng.choice(kinds) for _ in range(nt)) + ("#",) * nn)
        for sh, own in rng.sample([("b", None), ("n", None)] + [("t", k) for k in kinds], 2 + len(kinds)):
            L.forms[form] = [{"b": "yes", "n": "1", "t": "x1"}[sh]]
            L.own[form] = own
            progs = L.queries(form, types)
            if progs:
                ask[form] = rng.choice(progs[:max(1, len(progs) // 2)])      # the simpler half
                break
    return (rec, ask, names) if ask else None


def play(rng, rec, ask, names, episodes=20):
    """episodes of the made-up task, the answers by the ground"""
    out = []
    for _ in range(episodes):
        s, ep, met = records.State(), [], []
        target, queries = rng.randint(4, 12), rng.randint(1, 4)

        def row(form):
            things = []
            for k in form[2]:
                if k != "#":
                    old = [x for x in met if x[0] == k and x not in things]
                    pool = old if old and rng.random() < 0.6 else [x for x in names[k] if x not in things]
                    if not pool:
                        return None
                    things.append(rng.choice(pool))
            return tuple(things), tuple(rng.randint(1, 9) for k in form[2] if k == "#")
        tries = 0
        while (len([r for r in ep if not r[1]]) < target or queries) and tries < 80:
            tries += 1
            if queries and ep and rng.random() < 0.35:
                form = rng.choice(sorted(ask))
                got = row(form)
                if got is None:
                    continue
                p = ask[form]
                try:
                    a = records.answer(s, p[:-1] + (dict(p[-1]),) if p[0] in records.SPACE else p, got[0] + (form[0], records.ME), got[1], {})
                except (KeyError, IndexError, ValueError, TypeError):
                    a = None
                if a is None:
                    continue
                ep.append((form[0], True, got[0], got[1], form[2], ("yes" if a else "no") if a.__class__ is bool else a))
                queries -= 1
            else:
                form = rng.choice(sorted(rec))
                got = row(form)
                if got is None:
                    continue
                s.now = len(ep)
                t = got[0] + (form[0], records.ME)
                for x in got[0]:
                    s.born.setdefault(x, s.now)
                    met.append(x)
                try:
                    for name, k, pick, n in rec[form]:
                        records.MOVES[name][0](s, k, records.taken(t, got[1], pick), (got[1][n],) if n is not None else ())
                except (KeyError, IndexError, ValueError, TypeError):
                    continue
                ep.append((form[0], False, got[0], got[1], form[2], None))
        out.append(ep)
    return out


def dreams(n, seed=0):
    """n made-up tasks -> what prior.train takes: for every kind of record its possible readings as numbers and the true one"""
    rng, out, made = random.Random(seed), [], 0
    while made < n:
        task = make(rng)
        if task is None:
            continue
        rec, ask, names = task
        eps = play(rng, rec, ask, names)
        golds = [r[5] for ep in eps for r in ep if r[1]]
        if len(golds) < 20 or len(set(golds)) < 2:               # nothing is asked, or the answer is always the same
            continue
        m = records.Learner(eps)
        for f, p in rec.items():
            if f not in m.rec:
                continue
            see, tk, back = prior.look(m)[f]
            every = prior.shapes(m, f)
            hits = [i for i, sh in enumerate(every) if sh == prior.shape(p)]
            if hits:
                out.append((f"dream{made}", np.stack([np.concatenate([see, prior.said(sh, f, tk, back)]) for sh in every]), hits))
        made += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", default="", help="solved tasks to measure on: folder:results file, comma-separated")
    ap.add_argument("--out", default="", help="write the network taught on the dreams and on the solved tasks here")
    ap.add_argument("--keep", type=int, default=8)
    a = ap.parse_args()
    made = dreams(a.tasks, a.seed)
    print(f"{a.tasks} made-up tasks, {len(made)} kinds of rows", flush=True)
    real = {}
    for spec in [x for x in a.real.split(",") if x]:
        folder, results = spec.split(":")
        real[os.path.basename(folder.rstrip("/"))] = prior.examples(folder, results)
    every = [e for ex in real.values() for e in ex]
    if every:
        p = prior.place(prior.train(made, epochs=80), every)
        print(f"taught on the dreams alone: on {len(every)} real kinds of rows the reading found is the first for {100 * np.mean([x == 1 for x in p]):.0f}%, "
              f"among the first 8 for {100 * np.mean([x <= 8 for x in p]):.0f}%, among the first 24 for {100 * np.mean([x <= 24 for x in p]):.0f}%", flush=True)
        both, alone = [], []
        for name in real:                                        # every set by networks that have not seen it: with dreams and without
            others = [e for other, ex in real.items() if other != name for e in ex]
            both += prior.place(prior.train(others * 3 + made, epochs=80), real[name])
            alone += prior.place(prior.train(others), real[name])
        for label, p in (("solved tasks of the other sets", alone), ("the same and the dreams", both)):
            print(f"taught on {label}: the first for {100 * np.mean([x == 1 for x in p]):.0f}%, among the first 8 for {100 * np.mean([x <= 8 for x in p]):.0f}%, "
                  f"among the first 24 for {100 * np.mean([x <= 24 for x in p]):.0f}%", flush=True)
    if a.out:
        w = prior.train(every * 3 + made, epochs=80)
        json.dump({"weights": w, "keep": a.keep, "taught on": sorted(real) + [f"{a.tasks} made-up tasks"], "rows": len(every) + len(made)}, open(a.out, "w"))
        print(a.out, "written")


if __name__ == "__main__":
    main()
