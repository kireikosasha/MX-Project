#!/usr/bin/env python3
"""
A small network that says which reading of a record is worth trying first.

records.py finds what a record means by trying every move the ground has. That is sure, and slow: most of the moves
tried are far from anything this kind of row could mean. Here a small network looks at a kind of row before anything
is understood - how many things and numbers it has, whether its things are of one kind, whether they come back with
other partners, what the queries ask for - and at a move, and says how likely this move is the reading. It learns
that from tasks already solved (the readings records.py found). The learner then tries only the few moves the network
puts first; the ground checks them as before, and if the episodes are not explained this way, the whole search is run.
The network proposes, the ground decides: a wrong hint costs time, never an answer.

    python prior.py --tasks new/N:results/N.txt,new/M:results/M.txt --out prior.json     # learn from solved tasks
    python prior.py --tasks ... --apart                                                    # each set by a network that has not seen it
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

import records

NAMES = ["set", "unset", "rise", "swap", "moveall", "swapall", "pair", "unpair", "setat", "claim", "wait", "back", "copy", "tie", "untie", "tieboth", "tie2", "untie2", "cut", "ecopy", "eunion", "einter", "qset", "qadd", "qsub", "qpass", "qsame", "qmin", "qmax", "qsum", "qnot", "link", "tick", "clock"]   # the moves the network knows (those of records.py when it was taught)
WHO = (0, 1, 2, 3, -1, -2, "#", None)                        # a thing of the row by its place, the one not named, the operation, a number, nothing
ALL = {sort: n for sort, n in records.K.items()}


def shape(prog):
    """a reading of a row without its slots: ((move, the things it takes), ...)"""
    return tuple((name, pick) for name, k, pick, n in prog)


def look(m):
    """what is seen of every kind of record before anything is understood -> {form: numbers}"""
    if hasattr(m, "seen"):
        return m.seen
    quest = [f for f in m.forms if f[1]]
    recs = [f for f in m.forms if not f[1]]
    rows = sum(len(m.forms[f]) for f in recs) or 1
    asked = {m.own.get(f) for f in quest}                        # the kinds of the things that are answers
    first = {([k for k in f[2] if k != "#"] or [None])[0] for f in quest}
    golds = [g for f in quest for g in m.forms[f]]
    kinds = Counter(records.shape(g) for g in golds)
    total = sum(kinds.values()) or 1
    back, vals = {}, {}
    for ep in m.episodes:
        had = {}
        for op, q, things, numbers, ks, gold in ep:
            if q:
                continue
            f = (op, q, ks)
            vals.setdefault(f, []).extend(numbers)
            for i, j in ((0, 1), (1, 0)):
                if len(things) > max(i, j):
                    s = back.setdefault((f, i, j), [0, 0])       # a thing met again; met again with a new partner
                    was = had.setdefault((f, i, things[i]), set())
                    if was:
                        s[0] += 1
                        s[1] += things[j] not in was
                    was.add(things[j])
    m.seen = {}
    for f in recs:
        tk = [k for k in f[2] if k != "#"]
        nt, nn = len(tk), len(f[2]) - len(tk)
        v = [float(nt == i) for i in range(5)] + [float(nn == i) for i in range(3)]
        v += [float(len(tk) > j and tk[i] == tk[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        v += [float(len(tk) > i and tk[i].startswith("=")) for i in range(3)]
        v += [len(m.forms[f]) / rows, len(recs) / 10, sum(g[2] == f[2] and g != f for g in recs) / 5]
        v += [float(len(tk) > i and tk[i] in asked) for i in range(3)] + [float(len(tk) > i and tk[i] in first) for i in range(3)]
        for i, j in ((0, 1), (1, 0)):
            s = back.get((f, i, j), [0, 0])
            v += [s[0] / max(1, len(m.forms[f])), s[1] / max(1, s[0])]
        ns = vals.get(f, [])
        v += [np.log1p(sum(ns) / len(ns)) / 5 if ns else 0.0, float(bool(ns) and len(set(ns)) <= 6), float(bool(ns) and max(ns) <= 4)]
        v += [kinds["b"] / total, kinds["n"] / total, sum("," in g for g in golds) / total]
        m.seen[f] = (np.array(v, dtype=np.float32), tk, back)
    return m.seen


def said(sh, f, tk, back):
    """a reading as numbers: its moves, the things they take, and how the things of the first move go together"""
    v = []
    for part in range(2):
        name, pick = sh[part] if len(sh) > part else (None, ())
        v += [float(name == x) for x in NAMES]
        for place in range(3):
            who = (pick[place] if pick[place] > -10 else "#") if len(pick) > place else None
            v += [float(who == x) for x in WHO]
    pick = sh[0][1] if sh else ()
    i, j = (pick + (None, None))[:2]
    two = i is not None and j is not None and 0 <= i < len(tk) and 0 <= j < len(tk)
    s = back.get((f, i, j), [0, 0]) if two else [0, 0]
    v += [float(not sh), float(two and tk[i] == tk[j]), s[1] / max(1, s[0]) if two else 0.0, float(two and s[0] > 0)]
    return np.array(v, dtype=np.float32)


def shapes(m, f):
    """the readings a kind of row may have, without slots"""
    return list(dict.fromkeys([shape(p) for p in m.rows(*records.size(f), ALL, "FQEL")] + [()]))


class Hint:
    """the network at work: of the moves a row may be, the few to try"""

    def __init__(self, path, keep=None):
        w = json.load(open(path))
        self.w = [np.array(x, dtype=np.float32) for x in w["weights"]]
        self.most = keep or w["keep"]

    def score(self, x):
        h = np.maximum(x @ self.w[0] + self.w[1], 0)
        h = np.maximum(h @ self.w[2] + self.w[3], 0)
        return (h @ self.w[4] + self.w[5])[:, 0]

    def keep(self, m, u, cands):
        """of the candidate readings of the rows u, those whose shape the network puts among the first"""
        if not cands:
            return cands
        f = u[0]
        if not hasattr(m, "kept"):
            m.kept = {}
        if f not in m.kept:
            see, tk, back = look(m)[f]
            every = shapes(m, f)
            s = self.score(np.stack([np.concatenate([see, said(sh, f, tk, back)]) for sh in every]))
            m.kept[f] = {every[i] for i in np.argsort(-s)[:getattr(m, "most", None) or self.most]}
        return [c for c in cands if shape(c) in m.kept[f]]


def readings(path):
    """a results file of records.py --show -> {task: {(operation, kinds): reading without slots}}"""
    out, k = {}, None
    for line in open(path, encoding="utf-8"):
        t = re.match(r"task +(\d+):", line)
        if t:
            k = int(t.group(1))
            out[k] = {}
        r = re.match(r"^    (\S+) (.*?)\s+((?:\w+ [FQELTN]\d of \([^)]*\)(?:; )?)+|-)\s*$", line)
        if r and k is not None:
            moves = [] if r.group(3) == "-" else re.findall(r"(\w+) [FQELTN]\d of \(([^)]*)\)", r.group(3))
            out[k][(r.group(1), tuple(r.group(2).split()))] = tuple((name, tuple(int(x) for x in pick.replace(" ", "").split(",") if x)) for name, pick in moves)
    return out


def examples(folder, results, stories=20):
    """solved tasks -> [(set and task, the readings a row may have as numbers, which of them is the one found)]"""
    out = []
    for k, found in sorted(readings(results).items()):
        files = glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))
        if not files:
            continue
        m = records.Learner(records.read(files[0], stories))
        for f in m.rec:
            if (f[0], f[2]) not in found:
                continue
            see, tk, back = look(m)[f]
            every = shapes(m, f)
            right = found[(f[0], f[2])]
            hits = [i for i, sh in enumerate(every) if sh == right] or [i for i, sh in enumerate(every) if len(sh) == 1 and sh[0] in right]
            if hits:
                out.append((f"{os.path.basename(folder.rstrip('/'))}{k}", np.stack([np.concatenate([see, said(sh, f, tk, back)]) for sh in every]), hits))
    return out


def train(data, epochs=300, seed=0):
    """-> the weights; for every kind of row the network should put the reading found above the others"""
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(data[0][1].shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    xs = [(torch.tensor(x), hits) for _, x, hits in data]
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0
        for x, hits in xs:
            logp = torch.log_softmax(net(x)[:, 0], 0)
            loss = loss - torch.logsumexp(logp[hits], 0)
        (loss / len(xs)).backward()
        opt.step()
    return [p.detach().numpy().T.tolist() if p.dim() == 2 else p.detach().numpy().tolist() for p in net.parameters()]


def place(weights, data):
    """where the network puts the reading found: its place among the readings of every row (1 is the first)"""
    h = Hint.__new__(Hint)
    h.w = [np.array(x, dtype=np.float32) for x in weights]
    out = []
    for _, x, hits in data:
        order = list(np.argsort(-h.score(x)))
        out.append(1 + min(order.index(i) for i in hits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="solved tasks: folder:results file of records.py --show, comma-separated")
    ap.add_argument("--out", default="")
    ap.add_argument("--keep", type=int, default=8, help="how many readings of a row the learner tries")
    ap.add_argument("--apart", action="store_true", help="measure on every set with a network taught on the other sets")
    a = ap.parse_args()
    sets = {}
    for spec in a.tasks.split(","):
        folder, results = spec.split(":")
        sets[os.path.basename(folder.rstrip("/"))] = examples(folder, results)
        print(f"{folder}: {len(sets[os.path.basename(folder.rstrip('/'))])} kinds of rows with a reading", flush=True)
    if a.apart:
        for name in sets:
            w = train([e for other, ex in sets.items() if other != name for e in ex])
            p = place(w, sets[name])
            print(f"{name}: taught on the other sets, the reading found is the first for {100 * sum(x == 1 for x in p) / len(p):.0f}% of the kinds of rows, "
                  f"among the first {a.keep} for {100 * sum(x <= a.keep for x in p) / len(p):.0f}%, of {np.mean([len(x) for _, x, _ in sets[name]]):.0f} readings on average", flush=True)
    if a.out:
        data = [e for ex in sets.values() for e in ex]
        w = train(data)
        p = place(w, data)
        json.dump({"weights": w, "keep": a.keep, "taught on": sorted(sets), "rows": len(data)}, open(a.out, "w"))
        print(f"{a.out}: taught on {len(data)} kinds of rows; on them the reading found is among the first {a.keep} for {100 * sum(x <= a.keep for x in p) / len(p):.0f}%")


if __name__ == "__main__":
    main()
