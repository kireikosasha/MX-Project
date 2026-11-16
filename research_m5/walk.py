#!/usr/bin/env python3
"""
A walk over links read from a story, on two outside benchmarks: StepGame (Shi, Zhang, Lipani, 2022) and CLUTRR
(Sinha et al., 2019).

A story of StepGame says where things stand against each other ("Y is at a 45 degree angle to I, in the lower
righthand corner.") in some two hundred ways; the question asks how one thing stands against another, which
takes up to ten such sentences one after another, among sentences that do not matter. The models of the paper are
taught on 50 000 stories of one to five sentences and tried on ten thousand stories of every number up to ten.
A story of CLUTRR is a chain of kin ("the father of", "the sister of"...), the question is what the last is to
the first; taught on chains of two and three links, tried on chains of up to ten.

Here nothing is said of space or of kin. A story is links between things, every link of some kind (in StepGame:
the sentence with the names taken out); the question is answered by a walk from one thing to the other along the
links (the shortest way, found exactly). How the links of the way join is learnt, and there are two ways to join:

  register  of every kind of link it is learnt what it adds to a register - a few whole numbers (two by default),
            each of them -1, 0 or 1 more - softly; a link gone backwards takes away. What is said is learnt too: a
            table from the signs of the numbers of the register to the words of the answers. The state of the walk
            is how much is believed of everything the register may hold; the gradient of the answers goes through
            the walk. Then the choices are made hard, the walk is run on whole numbers, and every choice is tried
            over against the stories taught on (exactly: one choice changed, the stories counted).
  table     two links one after another make a relation that has a name - the answer of a story of two links; a
            named relation and a link make a named relation again (from the stories of three links). The table is
            learnt by counting; the way is joined by a chart - every stretch of it, in every order of joining that
            the table knows.

Which of the two a benchmark takes is not told: both are taught on four fifths of the stories and the one that
explains more of the fifth kept aside is kept.

    python walk.py --data .../StepGame/Dataset/TrainVersion
    python walk.py --data ... --stories 300 --least 2         # taught on 300 stories, none answered by one sentence
    python walk.py --clutrr .../gen23                         # the files gen23_train.csv and gen23_test.csv
    python walk.py --data ... --hold 0.2                      # a fifth of the kinds of sentences never shown: read from their words,
                                                              # and from stories without answers, by the register closing
"""
import argparse
import ast
import csv
import itertools
import json
import os
import random
import re
import time
from collections import Counter, deque

import torch

def steps_of(numbers):
    """what a link may add to a register of so many numbers; the last of them is the first gone backwards, and so on"""
    return list(itertools.product((-1, 0, 1), repeat=numbers))


def masked(sentence):
    """a sentence -> its kind (the names taken out: <0> is the name said first), the names"""
    names, out = [], []
    for w in re.findall(r"[A-Z](?='s)|[A-Za-z0-9:]+|'s|[.,]", sentence):
        if re.fullmatch(r"[A-Z]", w):
            if w not in names:
                names.append(w)
            out.append(f"<{names.index(w)}>")
        else:
            out.append(w.lower())
    return " ".join(out), names


def way(links, a, b):
    """the shortest way from a to b along the links [(kind, first name, second name)] -> [(kind, +1 or -1)], +1: the
    step goes from the second name of the sentence to the first. None: there is no way"""
    near = {}
    for kind, x, y in links:
        near.setdefault(y, []).append((x, kind, 1))
        near.setdefault(x, []).append((y, kind, -1))
    back, queue = {a: None}, deque([a])
    while queue:
        at = queue.popleft()
        if at == b:
            break
        for to, kind, sign in near.get(at, ()):
            if to not in back:
                back[to] = (at, kind, sign)
                queue.append(to)
    if b not in back:
        return None
    out = []
    while back[b] is not None:
        b, kind, sign = back[b]
        out.append((kind, sign))
    return out[::-1]


def read(path, kinds, asks, answers, grow, only=None):
    """a file of StepGame -> [(the way: [(kind, sign)], the kind of the question, the answer)]; a story the way of
    which cannot be found (a sentence of a kind not met in teaching, no way) has None for its way. only: the stories
    of these numbers in the file"""
    data = json.load(open(path))
    out = []
    for i, key in enumerate(data):
        if only is not None and i not in only:
            continue
        story, links = data[key], []
        for sentence in story["story"]:
            kind, names = masked(sentence)
            if len(names) == 2:
                if kind not in kinds and grow:
                    kinds[kind] = len(kinds)
                links.append((kinds.get(kind), names[1], names[0]))
        ask, names = masked(story["question"])
        if grow:
            asks.setdefault(ask, len(asks))
            answers.setdefault(story["label"], len(answers))
        steps = way([(k, y, x) for k, y, x in links], names[1], names[0]) if len(names) == 2 else None
        if steps is not None and any(k is None for k, _ in steps):
            steps = None
        out.append((steps, asks.get(ask), answers.get(story["label"])))
    return out


def told(links, a, b):
    """the links in the order they are told, if so they lead from a to b (each from where the one before came to; a
    thing may be come to twice) -> the way, or None"""
    out, at = [], a
    for kind, x, y in links:
        if y != at and x != at:
            return None
        out.append((kind, 1 if y == at else -1))
        at = x if y == at else y
    return out if at == b and out else None


def kin(path, kinds, asks, answers, grow, short=False):
    """a file of CLUTRR -> [(the way, the kind of the question, the answer, how many links the story has)]: the links
    are given as they are (who, the kind of kin, whom), the question is what the last is to the first. The way: the
    links as they are told, if they lead from the one to the other, else (or short) the shortest"""
    out = []
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        edges, types, ask = (ast.literal_eval(row[x]) for x in ("story_edges", "edge_types", "query_edge"))
        if grow:
            for t in types:
                kinds.setdefault(t, len(kinds))
            asks.setdefault("", len(asks))
            answers.setdefault(row["target_text"], len(answers))
        links = [(kinds.get(t), y, x) for (x, y), t in zip(edges, types)]
        steps = (None if short else told(links, ask[0], ask[1])) or way(links, ask[0], ask[1])
        if steps is not None and any(k is None for k, _ in steps):
            steps = None
        out.append((steps, 0, answers.get(row["target_text"]), len(types)))
    return out


class Table:
    """the other way for links to join: a table learnt by counting, the way joined by a chart"""

    def __init__(self, taught, names, answers):
        self.names, self.words, self.pairs = names, dict(answers), {}
        for steps, ask, gold in taught:                          # two links make what the story of the two is answered with
            if len(steps) == 2:
                self.count(ask, self.state(steps[0]), self.state(steps[1]), gold)
        for steps, ask, gold in taught:                          # three links: two of them joined by what is known already
            if len(steps) == 3:
                a, b, c = (self.state(x) for x in steps)
                for left in self.best(ask, a, b):
                    self.count(ask, ("name", left), c, gold)
                for right in self.best(ask, b, c):
                    self.count(ask, a, ("name", right), gold)

    def count(self, ask, a, b, gold):
        self.pairs.setdefault((ask, a, b), Counter())[gold] += 1

    def state(self, link):
        """a link -> itself, or the named relation, if the kind of the link is called by the word of an answer"""
        kind, sign = link
        word = self.names.get(kind)
        return ("name", self.words[word]) if sign > 0 and word in self.words else ("link", kind, sign)

    def best(self, ask, a, b):
        c = self.pairs.get((ask, a, b))
        return [c.most_common(1)[0][0]] if c else []

    def hard(self, steps, ask):
        """the chart: of every stretch of the way what named relations it may make and how much stands for each"""
        n = len(steps)
        span = {(i, i + 1): {self.state(steps[i]): 1.0} for i in range(n)}
        for width in range(2, n + 1):
            for i in range(n - width + 1):
                got = {}
                for m in range(i + 1, i + width):
                    for x, wx in span[(i, m)].items():
                        for y, wy in span[(m, i + width)].items():
                            c = self.pairs.get((ask, x, y))
                            for gold, times in (c or {}).items():
                                got[("name", gold)] = got.get(("name", gold), 0) + wx * wy * times / sum(c.values())
                whole = sum(got.values())
                span[(i, i + width)] = {k: v / whole for k, v in got.items()}
        top = {k: v for k, v in span[(0, n)].items() if k[0] == "name"}
        return max(top, key=top.get)[1] if top else None


class Walk(torch.nn.Module):
    """learnt: of every kind of sentence what it adds to the register; of every kind of question what is said by the
    signs of the register"""

    def __init__(self, kinds, asks, answers, seed=0, numbers=2):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.offs = steps_of(numbers)
        self.adds = torch.nn.Parameter(0.1 * torch.randn((kinds, len(self.offs)), generator=gen))
        self.says = torch.nn.Parameter(0.1 * torch.randn((asks, len(self.offs), answers), generator=gen))

    def forward(self, ways, asks):
        """softly: ways [(kind, sign)] of a batch, the kinds of their questions -> of every answer how much it is believed"""
        far, n = max(len(w) for w in ways), len(self.offs[0])
        dims = tuple(range(1, n + 1))
        p = torch.zeros((len(ways),) + (2 * far + 1,) * n)
        p[(slice(None),) + (far,) * n] = 1                       # the register: how much is believed of everything it may hold
        w = torch.softmax(self.adds, -1)
        for i in range(far):
            on = torch.tensor([i < len(x) for x in ways])
            kind = torch.tensor([x[i][0] if i < len(x) else 0 for x in ways])
            sign = torch.tensor([x[i][1] if i < len(x) else 1 for x in ways])
            add = torch.where(sign[:, None] > 0, w[kind], w[kind].flip(-1))
            new = sum(add[(slice(None), j) + (None,) * n] * torch.roll(p, off, dims) for j, off in enumerate(self.offs))
            p = torch.where(on[(slice(None),) + (None,) * n], new, p)
        for d in dims:                                           # of every number: below nought, nought, above
            p = p.transpose(1, d)
            p = torch.stack([p[:, :far].sum(1), p[:, far], p[:, far + 1:].sum(1)], 1).transpose(1, d)
        return (p.reshape(len(ways), -1)[:, :, None] * torch.softmax(self.says[asks], -1)).sum(1)

    def harden(self):
        """the choices made hard: of every kind of sentence what it adds, of every kind of question what is said"""
        self.add, self.say = self.adds.argmax(-1).tolist(), self.says.argmax(-1).tolist()

    def signs(self, steps):
        """the walk on whole numbers -> the signs of the register at its end (as one number)"""
        out = 0
        for d in range(len(self.offs[0])):
            x = sum(sign * self.offs[self.add[kind]][d] for kind, sign in steps)
            out = 3 * out + (x > 0) - (x < 0) + 1
        return out

    def hard(self, steps, ask):
        return self.say[ask][self.signs(steps)]

    def polish(self, taught, rounds=4, most=400):
        """every hard choice tried over against the stories taught on, one at a time, exactly; -> how many were changed"""
        by, changed = {}, 0
        for story in taught:
            for kind in {k for k, _ in story[0]}:
                by.setdefault(kind, []).append(story)
        for _ in range(rounds):
            before = changed
            for ask in {q for _, q, _ in taught}:                # what is said: of every state of the signs the commonest answer
                seen = {}
                for s, q, g in taught:
                    if q == ask:
                        seen.setdefault(self.signs(s), Counter())[g] += 1
                for signs, c in seen.items():
                    if self.say[ask][signs] != c.most_common(1)[0][0]:
                        self.say[ask][signs], changed = c.most_common(1)[0][0], changed + 1
            for kind, stories in by.items():                     # what a kind of sentence adds
                stories, was, score = stories[:most], self.add[kind], []
                for j in range(len(self.offs)):
                    self.add[kind] = j
                    score.append(sum(self.hard(s, q) == g for s, q, g in stories))
                self.add[kind] = was if score[was] == max(score) else score.index(max(score))
                changed += self.add[kind] != was
            if changed == before:
                break
        return changed


def register(taught, kinds, asks, answers, a, rng, log):
    """the register taught on the stories: several beginnings, the one that explains the stories best is kept"""
    best, long = None, [x for x in taught if len(x[0]) > 1]
    for seed in range(a.seed, a.seed + a.begin):                 # (begun anew several times: about one beginning in four ends in choices
        model = Walk(kinds, asks, answers, seed, a.numbers)      # that do not add up; the exact count on the stories taught on tells)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for step in range(a.steps):
            few = long if step < a.steps // 2 and len(long) >= 20 else taught        # (first the stories that take several sentences:
            batch = rng.sample(few, min(256, len(few)))                              # they say how the sentences add up)
            p = model([x[0] for x in batch], torch.tensor([x[1] for x in batch]))
            loss = -torch.log(p[torch.arange(len(batch)), torch.tensor([x[2] for x in batch])].clamp_min(1e-6)).mean()
            w = torch.softmax(model.adds, -1)                    # (at first the choices are kept from hardening too early)
            loss = loss + 0.5 * (1 - step / a.steps) * (w * torch.log(w.clamp_min(1e-9))).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.harden()
        first = sum(model.hard(s, q) == g for s, q, g in taught) / len(taught)
        changed = model.polish(taught)
        right = sum(model.hard(s, q) == g for s, q, g in taught) / len(taught)
        log(f"begun from {seed}: loss {loss.item():.3f}; made hard, of the stories taught on right {100 * first:.2f}%; "
            f"every choice tried over ({changed} changed): {100 * right:.2f}%")
        if best is None or right > best[0]:
            best = (right, model)
        if right == 1:
            break
    log(f"kept: the beginning that explains {100 * best[0]:.2f}% of the stories taught on")
    return best[1]


class Reader(torch.nn.Module):
    """a small network that says from the words of a sentence what its kind adds to the register"""

    def __init__(self, vocab, choices, width=32):
        super().__init__()
        self.vocab = {w: i + 1 for i, w in enumerate(sorted(vocab))}
        self.embed = torch.nn.Embedding(len(self.vocab) + 1, 24, padding_idx=0)
        self.gru = torch.nn.GRU(24, width, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(2 * width, choices)

    def forward(self, texts):
        most = max(len(t) for t in texts)
        ids = torch.tensor([[self.vocab.get(w, 0) for w in t] + [0] * (most - len(t)) for t in texts])
        mask = torch.tensor([[1.0] * len(t) + [0.0] * (most - len(t)) for t in texts])[:, :, None]
        if self.training:                                        # (a word in seven is hidden; the places of the names never)
            names = sum(ids == self.vocab.get(w, -1) for w in ("<0>", "<1>")) > 0
            ids = torch.where((torch.rand(ids.shape) < 0.15) & ~names, torch.zeros_like(ids), ids)
        h, _ = self.gru(self.embed(ids))
        return self.out((h * mask).sum(1) / mask.sum(1))


def closings(model, path, kinds, held, only):
    """stories read WITHOUT their answers. Where a sentence of a kind never shown links two things and another way
    leads between them through kinds that are known, the register must come to the same by both: that says what the
    sentence adds -> {kind: how often each choice came out}"""
    data, votes = json.load(open(path)), {}
    for n, key in enumerate(data):
        if n not in only:
            continue
        links = []
        for sentence in data[key]["story"]:
            kind, names = masked(sentence)
            if len(names) == 2 and kind in kinds:
                links.append((kinds[kind], names[1], names[0]))
        for i, (kind, x, y) in enumerate(links):
            if kind in held:
                other = way(links[:i] + links[i + 1:], y, x)
                if other and not any(o in held for o, _ in other):
                    total = tuple(sum(sign * model.offs[model.add[o]][d] for o, sign in other) for d in range(len(model.offs[0])))
                    if total in model.offs:
                        votes.setdefault(kind, Counter())[model.offs.index(total)] += 1
    return votes


def unseen(a, rng, t0):
    """StepGame with a share of the kinds of sentences never shown: the stories that have one of them on the way are
    not taught on. The table of the kinds shown is learnt through the walk as ever. The kinds never shown are then
    read in two ways. From their words: a small network is taught to say of a sentence shown what the table says
    (five of them, their beliefs added). And from stories without answers (the first half of every file of the
    test, the answers not looked at): by the register closing on another way. Tried on the second half of every
    file of the test, apart: the stories of kinds shown only, and the stories with a kind never shown on the way"""
    kinds, asks, answers = {}, {}, {}
    every = [x for x in read(os.path.join(a.data, "train.json"), kinds, asks, answers, True) if x[0] is not None]
    names = {i: k for k, i in kinds.items()}
    held = set(rng.sample(sorted(names), int(a.hold * len(names))))
    taught = [x for x in every if not any(k in held for k, _ in x[0])]
    shown = sorted({k for s, _, _ in taught for k, _ in s})
    model = register(taught, len(kinds), len(asks), len(answers), a, rng, lambda line: None)
    print(f"{len(names)} kinds of sentences, {len(held)} of them never shown; taught on {len(taught)} stories of {len(every)}: the table explains "
          f"{100 * sum(model.hard(s, q) == g for s, q, g in taught) / len(taught):.2f}% of them ({time.time() - t0:.0f} s)", flush=True)
    texts, gold, belief = [names[i].split() for i in shown], torch.tensor([model.add[i] for i in shown]), 0
    for seed in range(5):
        torch.manual_seed(a.seed + seed)
        net = Reader({w for t in texts for w in t}, len(model.offs))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for step in range(400):
            net.train()
            loss = torch.nn.functional.cross_entropy(net(texts), gold)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            belief = belief + net([names[i].split() for i in sorted(held)]).softmax(-1)
    worded = dict(zip(sorted(held), belief.argmax(-1).tolist()))
    print(f"the readers of words: {sum(p.numel() for p in net.parameters())} weights each, taught on the {len(shown)} sentences shown ({time.time() - t0:.0f} s)")
    size = {k: len(json.load(open(os.path.join(a.data, f"qa{k}_test.json")))) for k in range(1, 11)}
    first = {k: set(rng.sample(range(size[k] // 2), min(a.unanswered // 10, size[k] // 2)) if a.unanswered else range(size[k] // 2)) for k in range(1, 11)}
    closed, t1, rounds = {}, time.time(), 0
    while True:                                                  # (a kind settled is known from then on and helps to close on others)
        votes, rounds = {}, rounds + 1
        for k in range(1, 11):
            for kind, c in closings(model, os.path.join(a.data, f"qa{k}_test.json"), kinds, held - set(closed), first[k]).items():
                votes.setdefault(kind, Counter()).update(c)
        new = {kind: c.most_common(1)[0][0] for kind, c in votes.items() if c.most_common(1)[0][1] >= max(3, 0.7 * sum(c.values()))}
        if not new:
            break
        closed.update(new)
        for i, choice in new.items():
            model.add[i] = choice
    print(f"{sum(len(x) for x in first.values())} stories read without their answers ({rounds} times over, {time.time() - t1:.0f} s): of the {len(held)} kinds never "
          f"shown settled by the register closing on another way (three closings and more, seven in ten alike): {len(closed)}\n")
    print("tried on the second half of the stories of the test: right, %, on the stories whose way has only kinds shown; on those with a kind never shown")
    print("on the way - the kinds never shown read from their words; the same - read by the register closing (the rest from their words)")
    alone = {}
    for k in range(1, 11):
        tried = [x for x in read(os.path.join(a.data, f"qa{k}_test.json"), kinds, asks, answers, False, set(range(size[k] // 2, size[k]))) if x[0] is not None]
        old = [x for x in tried if not any(kind in held for kind, _ in x[0])]
        new = [x for x in tried if any(kind in held for kind, _ in x[0])]
        got = []
        for reading in (worded, {**worded, **closed}):
            for i, choice in reading.items():
                model.add[i] = choice
            got.append(100 * sum(model.hard(s, q) == g for s, q, g in new) / max(1, len(new)))
            for s, q, g in new if k == 1 else ():
                alone.setdefault((len(got), s[0][0]), []).append(model.hard(s, q) == g)
        print(f"  k = {k:2d}: {100 * sum(model.hard(s, q) == g for s, q, g in old) / len(old):6.2f} ({len(old)} stories)   {got[0]:6.2f}   {got[1]:6.2f} ({len(new)} stories)", flush=True)
    for n, label in ((1, "from their words"), (2, "by the register closing")):
        wrong = sorted(names[i] for (m, i), v in alone.items() if m == n and sum(v) < 0.9 * len(v))
        print(f"\nof the {sum(m == n for m, _ in alone)} kinds never shown that stand alone in a story tried, read rightly {label}: "
              f"{sum(m == n for m, _ in alone) - len(wrong)}; wrongly:")
        for t in wrong:
            print("   ", t)


def settle(taught, kinds, asks, answers, a, rng, t0):
    """which way of joining links the stories take: both taught on four fifths of them, the fifth kept aside tells.
    -> its name, and the table taught on all the stories if it is the table"""
    mixed = rng.sample(taught, len(taught))
    aside, most = mixed[:len(mixed) // 5], mixed[len(mixed) // 5:]
    names = {i: k for k, i in kinds.items()}
    by = {"register": register(most, len(kinds), len(asks), len(answers), a, rng, lambda line: None), "table": Table(most, names, answers)}
    right = {name: 100 * sum(model.hard(s, q) == g for s, q, g in aside) / max(1, len(aside)) for name, model in by.items()}
    kept = max(right, key=right.get)
    print(f"of the fifth of the stories kept aside ({len(aside)}) explained: by the register {right['register']:.2f}%, by the table {right['table']:.2f}% "
          f"- kept: the {kept} ({time.time() - t0:.0f} s)", flush=True)
    return kept, Table(taught, names, answers) if kept == "table" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="the folder TrainVersion of StepGame")
    ap.add_argument("--clutrr", default="", help="the files of CLUTRR without their ends: <this>_train.csv, <this>_test.csv")
    ap.add_argument("--hold", type=float, default=0, help="StepGame: so great a share of the kinds of sentences is never shown and is read from words")
    ap.add_argument("--unanswered", type=int, default=0, help="with --hold: so many stories are read without answers (0: the first half of the test)")
    ap.add_argument("--short", action="store_true", help="CLUTRR: always the shortest way, not the links as they are told")
    ap.add_argument("--stories", type=int, default=0, help="taught on so many stories drawn at random (0: on all)")
    ap.add_argument("--least", type=int, default=1, help="taught only on stories the answer of which takes so many sentences and more")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--numbers", type=int, default=2, help="how many numbers the register holds")
    ap.add_argument("--begin", type=int, default=6, help="so many beginnings; the one that explains the stories taught on best is kept")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng, t0 = random.Random(a.seed), time.time()
    kinds, asks, answers = {}, {}, {}
    if a.hold:
        return unseen(a, rng, t0)
    if a.clutrr:
        taught = [x[:3] for x in kin(a.clutrr + "_train.csv", kinds, asks, answers, True, a.short) if x[0] is not None]
        print(f"taught on {len(taught)} stories: {len(kinds)} kinds of links, {len(answers)} answers", flush=True)
        kept, model = settle(taught, kinds, asks, answers, a, rng, t0)
        if kept == "register":
            model = register(taught, len(kinds), len(asks), len(answers), a, rng, print)
        else:
            print(f"the table: {len(model.pairs)} pairs counted; {time.time() - t0:.0f} s with everything", flush=True)
        tried = kin(a.clutrr + "_test.csv", kinds, asks, answers, False, a.short)
        print("\ntried on the stories of the test, by how many links the story has: right, %")
        total = []
        for k in sorted({x[3] for x in tried}):
            some = [x for x in tried if x[3] == k]
            total.append(100 * sum(s is not None and model.hard(s, q) == g for s, q, g, _ in some) / len(some))
            print(f"  {k:2d} links: {total[-1]:6.2f}   ({len(some)} stories)", flush=True)
        print(f"  mean over the numbers of links: {sum(total) / len(total):.2f}; over the stories: "
              f"{100 * sum(s is not None and model.hard(s, q) == g for s, q, g, _ in tried) / len(tried):.2f}")
        return
    keys = [i for i, x in enumerate(read(os.path.join(a.data, "train.json"), {}, {}, {}, True)) if x[0] is not None and len(x[0]) >= a.least]
    keys = set(rng.sample(keys, a.stories) if a.stories else keys)
    taught = [x for i, x in enumerate(read(os.path.join(a.data, "train.json"), kinds, asks, answers, True, keys)) if x[0] is not None]
    print(f"taught on {len(taught)} stories (the answer takes {a.least} sentences and more): {len(kinds)} kinds of sentences, {len(asks)} of questions, "
          f"{len(answers)} answers", flush=True)

    kept, model = settle(taught, kinds, asks, answers, a, rng, t0)
    if kept == "register":
        model = register(taught, len(kinds), len(asks), len(answers), a, rng, print)
        print(f"{sum(p.numel() for p in model.parameters())} numbers learnt; {time.time() - t0:.0f} s with everything", flush=True)

    said = {}                                                    # the kinds of sentences the stories use for two things: alone in a
    for s, q, g in read(os.path.join(a.data, "train.json"), kinds, asks, answers, False):    # story, the same sentence the same way round is answered differently
        if s is not None and len(s) == 1:
            said.setdefault(s[0], Counter())[g] += 1
    double = {k for (k, _), c in said.items() if max(c.values()) < 0.9 * sum(c.values())}
    name = {i: k for k, i in kinds.items()}
    print(f"kinds of sentences that the stories themselves use for two things ({len(double)}): " + "; ".join(sorted(name.get(k, "?") for k in double)) + "\n", flush=True)

    print("tried on the stories of the test, by how many sentences the story has (the paper's k): right, %; the same on the stories whose way goes through")
    print("none of the kinds used for two things; the share of the stories where no way was found (a kind of sentence not met in teaching, or a thing never named): counted as wrong")
    total, clear = [], []
    for k in range(1, 11):
        tried = read(os.path.join(a.data, f"qa{k}_test.json"), kinds, asks, answers, False)
        right = sum(s is not None and model.hard(s, q) == g for s, q, g in tried)
        pure = [(s, q, g) for s, q, g in tried if s is not None and not any(kind in double for kind, _ in s)]
        none = sum(s is None for s, _, _ in tried)
        total.append(100 * right / len(tried))
        clear.append(100 * sum(model.hard(s, q) == g for s, q, g in pure) / max(1, len(pure)))
        print(f"  k = {k:2d}: {total[-1]:6.2f}   {clear[-1]:6.2f} (on {100 * len(pure) / len(tried):.0f}% of the stories)   no way: {100 * none / len(tried):.2f}%   "
              f"(the way is {sum(len(s) for s, _, _ in tried if s) / max(1, len(tried) - none):.1f} sentences long on the average)", flush=True)
    print(f"  mean:   {sum(total) / len(total):6.2f}   {sum(clear) / len(clear):6.2f}")


if __name__ == "__main__":
    main()
