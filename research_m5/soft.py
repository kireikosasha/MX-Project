#!/usr/bin/env python3
"""
The ground of records as a layer taught by a gradient: the seam between the hard and the smooth, crossed.

recall.py finds by trying and exact checking which part every kind of statement plays (which memory it writes, from
which of its places) and what every kind of question reads. Here the same memories are a layer of a network: every
kind of statement holds a choice among the parts it may play, every kind of question a choice among the things it
may read, and while it learns the choices are soft - a statement plays all its parts at once, each as much as it is
believed in, and the memory is the blend. The answers are compared with the right ones, the gradient goes back
through the memory to the choices, all kinds of statements learn at once. Then the choices are made hard (the part
most believed in) and the task is checked exactly by recall.py, word for word: a soft memory is only how it learns,
what it has learnt is a hard routing.

Two memories: the value of a thing (the latest holds) and links between things. Nothing is said of the sentences.

    python soft.py --data new/K --tasks 1,4,10
"""
import argparse
import glob
import os
import re
import time

import torch

import recall
import rows

SLOTS = 3
VALUE = [None] + [(n, i, j) for n in ("set", "as") for i in range(SLOTS) for j in range(SLOTS) if i != j] \
    + [(n, i) for n in ("lift", "clear") for i in range(SLOTS)] + [("swap", i, j) for i in range(SLOTS) for j in range(i + 1, SLOTS)]
LINK = [None] + [("link", i, j) for i in range(SLOTS) for j in range(SLOTS) if i != j]
ONE, TWO = [(i,) for i in range(SLOTS)], [(i, j) for i in range(SLOTS) for j in range(SLOTS) if i != j]
ASKS = {"t": [("value", n, a) for n in ("value", "end", "second", "holders", "inside") for a in ONE] + [("link", n, a) for n in ("after", "before") for a in ONE],
        "b": [("value", n, a) for n in ("is", "isin", "same") for a in TWO] + [("value", n, a) for n in ("any", "known") for a in ONE] + [("link", "leads", a) for a in TWO],
        "n": [("value", n, a) for n in ("count", "countin") for a in ONE] + [("link", n, a) for n in ("countafter", "countbefore") for a in ONE]}


def fits(part, nt):
    return part is None or all(x < nt for x in part[1:] if x.__class__ is int)


class Stories:
    """the stories of a task as tensors: of every row its kind and the things in it; of a question the right answer"""

    def __init__(self, task):
        self.task = task
        self.shape = {}
        for f, q in task.asked.items():
            golds = task.golds(q)
            self.shape[q] = "b" if all(g in ("yes", "no") for g in golds) else "n" if all(g.isdigit() for g in golds) else "t"
        B, S = len(task.stories), max(len(s) for s in task.stories)
        index = []
        for story in task.stories:
            seen = {}
            for f, quest, things, numbers, gold in story:
                for x in things + tuple(x for x in (gold or "").split(",") if recall.thing(x)):
                    seen.setdefault(x, len(seen))
            index.append(seen)
        self.N = N = max(len(x) for x in index)
        self.said = torch.full((B, S), len(task.forms), dtype=torch.long)       # (the last number: not a statement)
        self.asked = torch.full((B, S), -1, dtype=torch.long)
        self.things = torch.zeros((B, S, SLOTS), dtype=torch.long)
        self.gold = torch.zeros((B, S, N))                                       # the things of the answer; yes; the number
        self.yes, self.number = torch.zeros((B, S)), torch.zeros((B, S))
        for b, story in enumerate(task.stories):
            for s, (f, quest, things, numbers, gold) in enumerate(story):
                for i, x in enumerate(things[:SLOTS]):
                    self.things[b, s, i] = index[b][x]
                if quest:
                    self.asked[b, s] = f
                    self.yes[b, s] = float(gold == "yes")
                    self.number[b, s] = float(gold) if gold.isdigit() else 0.0
                    for x in gold.split(","):
                        if recall.thing(x):
                            self.gold[b, s, index[b][x]] = 1.0
                else:
                    self.said[b, s] = f


class Layer(torch.nn.Module):
    """the memories and the choices: of every kind of statement the part it plays in each memory, of every kind of
    question what it reads"""

    def __init__(self, data, seed=0, quiet=1.0):
        super().__init__()
        task, gen = data.task, torch.Generator().manual_seed(seed)
        sizes = {n: sum(k != "#" for k in f[2]) for f, n in task.forms.items()}
        self.shape = data.shape
        self.nt = {q: sum(k != "#" for k in f[2]) for f, q in task.asked.items()}
        F = len(task.forms)
        self.maskv = torch.tensor([[0.0 if fits(p, sizes[f]) else -1e9 for p in VALUE] for f in range(F)] + [[0.0] + [-1e9] * (len(VALUE) - 1)])
        self.maskl = torch.tensor([[0.0 if fits(p, sizes[f]) else -1e9 for p in LINK] for f in range(F)] + [[0.0] + [-1e9] * (len(LINK) - 1)])
        self.value = torch.nn.Parameter(0.3 * torch.randn((F + 1, len(VALUE)), generator=gen))
        self.link = torch.nn.Parameter(0.3 * torch.randn((F + 1, len(LINK)), generator=gen))
        self.also = torch.nn.Parameter(0.3 * torch.randn((F + 1, len(VALUE)), generator=gen))       # a statement may say two things at once
        with torch.no_grad():                                    # (at first a statement rather plays no part, and a second one less still)
            self.value[:, 0] += quiet
            self.link[:, 0] += quiet
            self.also[:, 0] += 3 * quiet
        self.read = torch.nn.ParameterDict({str(q): torch.nn.Parameter(0.3 * torch.randn(len(ASKS[self.shape[q]]), generator=gen)) for q in self.nt})
        self.maskr = {q: torch.tensor([0.0 if all(x < self.nt[q] for x in a) else -1e9 for _, _, a in ASKS[self.shape[q]]]) for q in self.nt}
        self.turn = torch.nn.Parameter(torch.tensor([[4.0, 0.0] for _ in self.nt]))       # of a yes or no: which way, and what where nothing is known
        self.where = {q: i for i, q in enumerate(self.nt)}
        self.given = None                                        # the choices as someone else makes them (the reader of words)

    def parts(self):
        value, link, also = self.given[:3] if self.given else (self.value, self.link, self.also)
        return torch.softmax(value + self.maskv, -1), torch.softmax(link + self.maskl, -1), torch.softmax(also + self.maskv, -1)

    def reads(self, q):
        return torch.softmax((self.given[3][q] if self.given else self.read[str(q)]) + self.maskr[q], -1)

    def turned(self, q):
        return self.given[4][q] if self.given else self.turn[self.where[q]]

    def forward(self, data):
        """the stories gone through, every statement playing all its parts at once -> the loss; what the parts cost"""
        B, S, N = data.said.shape[0], data.said.shape[1], data.N
        pv, pl, pa = self.parts()
        V = torch.zeros((B, N, N + 1))
        V[:, :, N] = 1.0                                         # (nothing has a value yet)
        A = torch.zeros((B, N, N))
        ar, none = torch.arange(B), torch.nn.functional.one_hot(torch.tensor(N), N + 1).float()
        loss, count = 0.0, 0
        for s in range(S):
            f, t = data.said[:, s], data.things[:, s]
            if (f < pv.shape[0] - 1).any():
                for p in (pv[f], pa[f]):                         # (the second part after the first)
                    D = torch.zeros_like(V)
                    for o, part in enumerate(VALUE):
                        if part is None or not (p[:, o] > 1e-6).any():
                            continue
                        rows_, new = [t[:, part[1]]], None
                        cur = V[ar, rows_[0]]
                        if part[0] == "set":
                            new = [torch.nn.functional.one_hot(t[:, part[2]], N + 1).float()]
                        elif part[0] == "clear":
                            new = [none.expand(B, N + 1)]
                        elif part[0] == "as":
                            new = [V[ar, t[:, part[2]]]]
                        elif part[0] == "lift":
                            up = torch.bmm(cur[:, None, :N], V)[:, 0]
                            new = [up + cur[:, N:] * none]
                        else:                                        # swap
                            rows_.append(t[:, part[2]])
                            new = [V[ar, rows_[1]], cur]
                        for r, x in zip(rows_, new):
                            D[ar, r] = D[ar, r] + p[:, o, None] * (x - V[ar, r])
                    V = V + D
                p, D = pl[f], torch.zeros_like(A)
                for o, part in enumerate(LINK):
                    if part is not None and (p[:, o] > 1e-6).any():
                        a, b = t[:, part[1]], t[:, part[2]]
                        D[ar, a, b] = D[ar, a, b] + p[:, o] * (1 - A[ar, a, b])
                A = (A + D).clamp(0, 1)
            qs = data.asked[:, s]
            if (qs >= 0).any():
                Vt, Vn = V[:, :, :N], V[:, :, N]
                W = Vt + torch.diag_embed(Vn)
                M, hops = Vt, [Vt]
                for _ in range(2):
                    M = torch.bmm(M, Vt).clamp(0, 1)
                    hops.append(M)
                inside = 1 - (1 - hops[0]) * (1 - hops[1]) * (1 - hops[2])          # [x, y]: y is on the chain of x
                C = A
                for _ in range(3):
                    C = 1 - (1 - C) * (1 - torch.bmm(C, C).clamp(0, 1))
                for q in qs[qs >= 0].unique().tolist():
                    who = (qs == q).nonzero()[:, 0]
                    u, shape, said = t[who], self.shape[q], []
                    for (sort, name, at), ok in zip(ASKS[shape], self.maskr[q]):
                        if ok < 0:
                            said.append(torch.zeros((len(who), N)) if shape == "t" else torch.zeros(len(who)))
                            continue
                        a = u[:, at[0]]
                        b = u[:, at[1]] if len(at) > 1 else None
                        if name == "value":
                            x = Vt[who, a]
                        elif name == "second":
                            x = torch.bmm(Vt[who, a][:, None], Vt[who])[:, 0]
                        elif name == "end":
                            x = Vt[who, a]
                            for _ in range(3):
                                x = torch.bmm(x[:, None], W[who])[:, 0]
                        elif name == "holders":
                            x = Vt[who, :, a]
                        elif name == "inside":
                            x = inside[who, :, a]
                        elif name == "after":
                            x = C[who, a]
                        elif name == "before":
                            x = C[who, :, a]
                        elif name == "is":
                            x = Vt[who, a, b]
                        elif name == "isin":
                            x = inside[who, a, b]
                        elif name == "same":
                            x = (Vt[who, a] * Vt[who, b]).sum(-1)
                        elif name == "any":
                            x = 1 - (1 - Vt[who, :, a]).prod(-1)
                        elif name == "known":
                            x = 1 - Vn[who, a]
                        elif name == "leads":
                            x = C[who, a, b]
                        elif name == "count":
                            x = Vt[who, :, a].sum(-1)
                        elif name == "countin":
                            x = inside[who, :, a].sum(-1)
                        elif name == "countafter":
                            x = C[who, a].sum(-1)
                        else:
                            x = C[who, :, a].sum(-1)
                        said.append(x)
                    said, r = torch.stack(said, 1), self.reads(q)
                    if shape == "t":
                        told = (r[None, :, None] * said).sum(1).clamp(1e-5, 1 - 1e-5)
                        loss = loss + torch.nn.functional.binary_cross_entropy(told, data.gold[who, s], reduction="none").sum(-1).sum()
                    elif shape == "b":
                        w, c = self.turned(q)
                        told = (r[None] * torch.sigmoid(w * (2 * said - 1) + c)).sum(1).clamp(1e-5, 1 - 1e-5)
                        loss = loss + torch.nn.functional.binary_cross_entropy(told, data.yes[who, s], reduction="sum")
                    else:
                        loss = loss + 2.0 * (r[None] * (said - data.number[who, s][:, None]) ** 2).sum()
                    count += len(who)
        return loss / max(1, count), ((1 - pv[:-1, 0]).sum() + (1 - pl[:-1, 0]).sum() + (1 - pa[:-1, 0]).sum()) / max(1, count)

    def hard(self, task):
        """the choices made hard: of every kind of question the way of looking, what it asks and the parts of the kinds
        of statements in that memory; checked on the stories exactly -> {q: recall.Reading}"""
        pv, pl, pa = self.parts()
        two = pa[:-1].argmax(-1).tolist()
        roles = {"value": {f: ("two", VALUE[o], VALUE[two[f]]) if o and two[f] else VALUE[o or two[f]] for f, o in enumerate(pv[:-1].argmax(-1).tolist()) if o or two[f]},
                 "link": {f: LINK[o] for f, o in enumerate(pl[:-1].argmax(-1).tolist()) if o}}
        out = {}
        for q in self.nt:
            sort, name, at = ASKS[self.shape[q]][int(self.reads(q).argmax())]
            got = recall.grade(recall.run(task, q, sort, (name, at), roles[sort], task.own(q)))
            out[q] = recall.Reading(sort, (name, at), dict(roles[sort]), got[1], got[2], got[0], len(task.golds(q)))
        return out


def words(text, quest):
    """a kind of sentence as tokens: its words, and its places for things each by its number"""
    n, out = 0, ["?" if quest else "."]
    for w in text.split():
        if w == "*":
            out.append(f"<{min(n, SLOTS - 1)}>")
            n += 1
        else:
            out.append(w)
    return out


class Reader(torch.nn.Module):
    """a small network that reads the words of a kind of sentence and makes its choices: what part a statement plays
    in each memory, what a question reads. It is taught through the layer, by the answers of many tasks at once -
    what a word means is what it makes the memory do"""

    def __init__(self, vocab, width=48):
        super().__init__()
        self.vocab = {w: i + 1 for i, w in enumerate(sorted(vocab))}
        self.embed = torch.nn.Embedding(len(self.vocab) + 1, width)      # (0: a word never met)
        self.gru = torch.nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.heads = torch.nn.ModuleDict({"value": torch.nn.Linear(2 * width, len(VALUE)), "link": torch.nn.Linear(2 * width, len(LINK)),
                                          "also": torch.nn.Linear(2 * width, len(VALUE)), "turn": torch.nn.Linear(2 * width, 2),
                                          **{k: torch.nn.Linear(2 * width, len(v)) for k, v in ASKS.items()}})
        with torch.no_grad():
            self.heads["value"].bias[0] += 1.0
            self.heads["link"].bias[0] += 1.0
            self.heads["also"].bias[0] += 3.0

    def read(self, texts):
        ids = [[self.vocab.get(w, 0) for w in t] for t in texts]
        most = max(len(x) for x in ids)
        x = self.embed(torch.tensor([x + [0] * (most - len(x)) for x in ids]))
        out, _ = self.gru(torch.nn.utils.rnn.pack_padded_sequence(x, [len(i) for i in ids], batch_first=True, enforce_sorted=False))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = torch.tensor([[1.0] * len(i) + [0.0] * (most - len(i)) for i in ids])[:, :, None]
        return (out * mask).sum(1) / mask.sum(1)

    def choices(self, task, layer):
        """the choices for a task, read from the words of its kinds of sentences -> as Layer.given takes them"""
        said = sorted(task.forms.items(), key=lambda x: x[1])
        asked = sorted(task.asked.items(), key=lambda x: x[1])
        h = self.read([words(f[0], False) for f, _ in said] + [words(f[0], True) for f, _ in asked])
        hs, hq = h[:len(said)], h[len(said):]
        pad = lambda x: torch.cat([x, torch.zeros((1, x.shape[1]))])
        turn = self.heads["turn"](hq)
        return (pad(self.heads["value"](hs)), pad(self.heads["link"](hs)), pad(self.heads["also"](hs)),
                {q: self.heads[layer.shape[q]](hq[i]) for i, (_, q) in enumerate(asked)},
                {q: torch.stack([4.0 + turn[i, 0], turn[i, 1]]) for i, (_, q) in enumerate(asked)})


def learn(task, steps=300, seeds=3, lr=0.1, price=0.1, log=None, start=None, until=None):
    """the choices taught by the gradient from several beginnings; the one that, made hard, explains most is kept
    (a part costs a tenth of a wrong answer: a statement no question can tell anything of drifts to playing none;
    until: the time by which it must be done)
    -> the readings, how many questions they explain exactly, of how many"""
    data, best = Stories(task), None
    for seed in range(seeds):
        net = Layer(data, seed)
        if start is not None:                                    # (it begins from the choices the reader of words has made)
            with torch.no_grad():
                net.value.copy_(start[0]), net.link.copy_(start[1]), net.also.copy_(start[2])
                for q in net.nt:
                    net.read[str(q)].copy_(start[3][q])
                    net.turn[net.where[q]].copy_(start[4][q])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for step in range(steps):
            if until and time.time() > until:
                break
            loss, cost = net(data)
            opt.zero_grad()
            (loss + price * cost).backward()
            opt.step()
            if log and step % 50 == 0:
                log(f"    beginning {seed}, step {step}: loss {loss.item():.3f}")
        readings = net.hard(task)
        right = sum(r.right for r in readings.values())
        if best is None or right > best[1]:
            best = (readings, right)
        if right == sum(r.total for r in readings.values()) or until and time.time() > until:
            break
    return best[0], best[1], sum(r.total for r in best[0].values())


def polished(task, readings):
    """what the gradient has found, finished by the exact check: for a kind of question not explained in full the parts
    of the kinds of statements are tried one at a time from where they stand (recall.py) -> the readings"""
    out = {}
    for q, r in readings.items():
        if r.right < r.total:
            s = recall.Search(task, q, r.sort, r.ask, recall.parts(task, r.sort)[0], task.own(q), 2000)
            s.roles = dict(r.roles)
            s.score = recall.grade(recall.run(task, q, r.sort, r.ask, s.roles, task.own(q)))
            s.level([(f,) for f in s.mine])
            r = s.reading() if s.score[0] > r.right else r
        out[q] = r
    return out


def opened(folder, k, guess, stories):
    """a task with the guess about its words that is named -> the reader of rows.py, the task; None if there is no such guess"""
    raw = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], stories)
    for reader in rows.guesses(raw[:20]):
        if "; ".join(reader.guessed) == guess:
            return reader, recall.Task(reader.rows(raw), False)
    return None


def pool(results, folders, hold):
    """the tasks the layer has learnt by itself, or nearly (the files a run of this script wrote) -> [(set, task, guess)]"""
    out = []
    for name in folders:
        if name in hold or not os.path.exists(os.path.join(results, name + ".txt")):
            continue
        for line in open(os.path.join(results, name + ".txt")):
            m = re.match(r"task +(\d+): by the gradient from \d+ stories, (\d+)/(\d+) of their questions right when made hard, .*? s\)(?: - (.*))?$", line.rstrip("\n"))
            if m and 10 * int(m.group(2)) >= 9 * int(m.group(3)):         # (nine questions of ten right: its words were read rightly)
                out.append((name, int(m.group(1)), m.group(4) or ""))
    return out


def teach(tasks, a, log=print):
    """the reader of words taught through the layer on many tasks at once: at every step the choices of a task are
    read from its words, its stories gone through softly, and the answers pull at the reader -> the reader"""
    gen = torch.Generator().manual_seed(a.seed)
    vocab = {w for _, task, _, _ in tasks for f in list(task.forms) + list(task.asked) for w in words(f[0], False)} | {"?", "."}
    net = Reader(vocab)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0, mean = time.time(), None
    for step in range(a.teach):
        for group in opt.param_groups:                           # (the steps grow smaller to the end)
            group["lr"] = 2e-3 * (1 - 0.9 * step / a.teach)
        loss = 0.0
        for i in torch.randint(len(tasks), (4,), generator=gen).tolist():
            _, task, data, layer = tasks[i]
            layer.given = net.choices(task, layer)
            got, cost = layer(data)
            loss = loss + got + 0.1 * cost
        opt.zero_grad()
        loss.backward()
        opt.step()
        mean = loss.item() / 4 if mean is None else 0.98 * mean + 0.02 * loss.item() / 4
        if step % 200 == 0 or step == a.teach - 1:
            full = 0
            with torch.no_grad():
                for _, task, data, layer in tasks:
                    layer.given = net.choices(task, layer)
                    got = layer.hard(task)
                    full += all(r.right == r.total for r in got.values())
            log(f"step {step}, {(time.time() - t0) / 60:.1f} min: loss {mean:.3f}; of the {len(tasks)} tasks it is taught on, read from their words alone and explained in full: {full}")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--results", default="", help="to teach the reader of words: the folder of the files this script wrote for the sets (A.txt, ...)")
    ap.add_argument("--sets", default="", help="name=folder,... of the sets of --results")
    ap.add_argument("--hold", default="", help="sets the reader of words is not taught on")
    ap.add_argument("--teach", type=int, default=0, help="teach the reader of words for so many steps")
    ap.add_argument("--words", default="", help="the reader of words: where it is kept (with --teach), or the one to read new tasks with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--stories", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=6, help="a task is given so long at the most")
    ap.add_argument("--polish", action="store_true", help="what the gradient has found is finished by the exact check of recall.py")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.teach:
        sets = dict(x.split("=") for x in a.sets.split(","))
        tasks = []
        for name, k, guess in pool(a.results, sets, a.hold.split(",")):
            got = opened(sets[name], k, guess, a.stories)
            if got:
                data = Stories(got[1])
                tasks.append((f"{name}{k}", got[1], data, Layer(data)))
        print(f"{len(tasks)} tasks the layer has learnt by itself, to teach the reader of words on: " + " ".join(t[0] for t in tasks), flush=True)
        net = teach(tasks, a)
        torch.save({"vocab": sorted(net.vocab), "weights": net.state_dict()}, a.words)
        return
    reader_of_words = None
    if a.words:
        saved = torch.load(a.words)
        reader_of_words = Reader(saved["vocab"])
        reader_of_words.load_state_dict(saved["weights"])
    files = glob.glob(os.path.join(a.data, "qa*_train.txt"))
    tasks = [int(x) for x in a.tasks.split(",") if x] or sorted(int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1)) for f in files)
    for k in tasks:
        t0 = time.time()
        raw = rows.stories(glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0], a.stories)
        best, tried = None, []
        if reader_of_words is not None:                          # the choices read from the words alone, made hard and checked: at once;
            for reader in rows.guesses(raw[:20]):                # then the same choices as a beginning for the gradient (a third of the steps)
                task = recall.Task(reader.rows(raw), False)
                if max(sum(k != "#" for k in f[2]) for f in list(task.forms) + list(task.asked)) > SLOTS:
                    continue
                layer = Layer(Stories(task))
                with torch.no_grad():
                    layer.given = reader_of_words.choices(task, layer)
                    readings = layer.hard(task)
                right, total = sum(r.right for r in readings.values()), sum(r.total for r in readings.values())
                tried.append((right - 0.5 * len(readings) - 3 * bool(reader.guessed), reader, task, readings, right, total, layer.given))
                if right == total:
                    break
            if not tried:
                print(f"task {k:2d}: its sentences have more things than the layer has places for", flush=True)
                continue
            mark, reader, task, readings, right, total, given = max(tried, key=lambda t: t[0])
            once = (right, total)
            if right < total:
                readings, right, total = learn(task, a.steps // 3, 1, start=given)
            tried = [(mark, reader, task, readings, right, total)]
            tuned = right
            if right < total:
                readings = polished(task, readings)
                right = sum(r.right for r in readings.values())
            tried = [(mark, reader, task, readings, right, total)]
            print(f"task {k:2d}: read from its words at once {once[0]}/{once[1]}" + (" - taken at once" if once[0] == once[1] else
                  f"; after {a.steps // 3} steps of the gradient from there {tuned}/{total}" + ("" if tuned == total else f"; finished by the exact check {right}/{total}")), flush=True)
        for reader in rows.guesses(raw[:20]) if reader_of_words is None else ():      # the stories as they are, then with one guess about their words (the
            task = recall.Task(reader.rows(raw), False)          # guesses of rows.py; a guess is paid for as in recall.py): each
            if max(sum(k != "#" for k in f[2]) for f in list(task.forms) + list(task.asked)) > SLOTS:        # taught shortly
                continue
            readings, right, total = learn(task, a.steps // 3, 1, until=t0 + 30 * a.minutes)
            tried.append((right - 0.5 * len(readings) - 3 * bool(reader.guessed), reader, task, readings, right, total))
            if right == total:
                break
        for mark, reader, task, readings, right, total in sorted(tried, key=lambda t: -t[0])[:2]:      # the best two taught in full
            if right < total and reader_of_words is None:
                readings, right, total = learn(task, a.steps, a.seeds, log=print if a.show else None, until=t0 + 60 * a.minutes)
                mark = right - 0.5 * len(readings) - 3 * bool(reader.guessed)
            if best is None or mark > best[0]:
                best = (mark, reader, task, readings, right, total)
            if right == total:
                break
        if best is None:
            print(f"task {k:2d}: its sentences have more things than the layer has places for", flush=True)
            continue
        _, reader, task, readings, right, total = best
        if a.polish and right < total and reader_of_words is None:
            readings = polished(task, readings)
            print(f"task {k:2d}: by the gradient {right}/{total}, finished by the exact check {sum(r.right for r in readings.values())}/{total}", flush=True)
            right = sum(r.right for r in readings.values())
        m = recall.Learner.__new__(recall.Learner)               # (the learner of recall.py with nothing learnt: it only checks)
        m.task, m.readings = task, readings
        m.count = {q: len(task.golds(q)) for q in task.asked.values()}
        m.usual = {q: recall.Counter(task.golds(q)).most_common(1)[0][0] for q in task.asked.values()}
        m.own = {q: task.own(q) for q in task.asked.values()}
        full = sum(r.right == r.total and r.total >= 3 for r in m.readings.values())
        print(f"task {k:2d}: by the gradient from {a.stories} stories, {right}/{total} of their questions right when made hard, {full} of {len(m.readings)} kinds of "
              f"questions explained in full ({time.time() - t0:.1f} s)" + (" - " + "; ".join(reader.guessed) if reader.guessed else ""), flush=True)
        if a.show:
            back = {n: f for f, n in task.asked.items()}
            for q, r in sorted(m.readings.items()):
                print(f"  ? {back[q][0]}: {r.right}/{r.total} {r.tell(task)}"[:300], flush=True)
        tests = reader.rows(rows.stories(glob.glob(os.path.join(a.data, f"qa{k}_*test.txt"))[0]))
        r, t, sr, st = m.test(tests)
        print(f"test {k:2d}: {r}/{t} = {100 * r / max(1, t):.1f}%; sure of {full} of {len(m.readings)} kinds of questions, {sr}/{st} right on them", flush=True)


if __name__ == "__main__":
    main()
