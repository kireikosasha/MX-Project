#!/usr/bin/env python3
"""
One program for stories told in words about things and what links them.

Nothing is said to it of what the stories are about. A story names things (names in square brackets) in sentences;
a question asks what one thing is to another; the answer is a word. From stories with their answers the program
learns to read and to join what it reads, and then answers stories it has not seen, longer ones too. Its parts:

  memory     the things of a story and, of every two things named together, the sentences that name them: kept as
             they are, nothing blurred
  reading    what the sentences of such a pair say: one choice of a few. A small network says it from the words; a
             view met many times has besides a row of its own (it is remembered as it is)
  joining    two machines, both exact once taught:
               the register  a link adds -1, 0 or +1 to each of a few whole numbers, and going against it takes the
                             same away; the answer is told by the signs of the numbers at the end of the way
               the table     two links one after another make a relation that has a name; a way is joined by a chart
  learning   by the answers alone: the choices are soft and a gradient goes through the machine; then they are made
             hard and counted exactly on the stories taught on (text.py, walk.py). Which machine suits the stories
             is told by the fifth of them kept aside

    python one.py --stepgame .../StepGame/Dataset/TrainVersion
    python one.py --clutrr DIR            (gen23_train.csv and gen23_test.csv, as for text.py)
    python one.py --stories teach.jsonl --tried try.jsonl
                                          (a line: {"story": "... [Ann] ... [Bob] ...", "asked": ["Ann", "Bob"],
                                                    "question": "", "answer": "sister", "k": 3})
"""
import argparse
import ast
import copy
import csv
import json
import os
import random
import re
import time
from collections import Counter

import torch

import text
import walk

WORD = re.compile(r"\[[^\]]+\]|[A-Za-z0-9:]+|'s|[.,!?]")


def story(sentences, asked, question, answer, k):
    """the common form: sentences (a text is cut at its full stops; a list is taken sentence by sentence), the two
    asked about, the words of the question, the answer, a number to sort the results by -> (the sentences as words
    with names as numbers, the pairs near, {}, the two asked about as numbers, the answer, k, the kind of question)"""
    names, told = [], []
    for part in ([sentences] if sentences.__class__ is str else sentences):
        told.append([])
        for w in WORD.findall(part):
            if w[0] == "[":
                if w[1:-1] not in names:
                    names.append(w[1:-1])
                told[-1].append(names.index(w[1:-1]))
            else:
                told[-1].append(w.lower())
                if w in ".!?" and sentences.__class__ is str:
                    told.append([])
    told = [s for s in told if s]
    who = tuple(names.index(n) if n in names else None for n in asked)
    return told, text.near(told), {}, who, answer, k, question


def stepgame(folder):
    """StepGame -> the stories to teach on, {name: the stories to try}. The things are single capital letters"""
    def read(name, k):
        out = []
        for one in json.load(open(os.path.join(folder, name))).values():
            mark = lambda s: re.sub(r"\b([A-Z])\b", r"[\1]", s)
            asked = re.findall(r"\b[A-Z]\b", one["question"])[-2:]
            question = re.sub(r"\b[A-Z]\b", "<>", one["question"]).lower()
            out.append(story([mark(s) for s in one["story"]], asked, question, one["label"], k))
        return out
    return read("train.json", 0), {f"k={k}": read(f"qa{k}_test.json", k) for k in range(1, 11)}


def clutrr(folder, teach="gen23_train.csv", tests=("gen23_test.csv",)):
    """CLUTRR from its words alone: the story, the two asked about, the answer"""
    def read(name):
        out = []
        for row in csv.DictReader(open(os.path.join(folder, name), newline="", encoding="utf-8")):
            answer = row.get("target_text") or {"neice": "niece"}.get(row["target"], row["target"])
            out.append(story(row["story"], ast.literal_eval(row["query"]), "", answer, len(ast.literal_eval(row["edge_types"])) if "edge_types" in row else 0))
        return out
    return read(teach), {name: read(name) for name in tests}


def plain(path):
    return [story(x["story"], x["asked"], x.get("question", ""), x["answer"], x.get("k", 0)) for x in map(json.loads, open(path))]


def steps_of(one):
    """the shortest way between the two asked about, every link as (the view of the pair in the order the sentence
    names them, +1 if gone that way, -1 if against it) -> the way or None"""
    told, pairs, _, (x, y), *_ = one
    if x is None or y is None:
        return None
    together = text.together(told)
    way = text.surest(pairs, lambda p, q: 1 if (p, q) in together else 4, x, y)
    out = []
    for p, q in way or []:
        both = [s for s in told if p in s and q in s] or [s for s in told if p in s or q in s]
        first = next(w for s in both for w in s if w in (p, q))
        a, b = (p, q) if first == p else (q, p)
        out.append((tuple(text.view(told, a, b, True)), 1 if (a, b) == (p, q) else -1))
    return out or None


class Register:
    """the register taught through a reader: what a link adds is said from the words of its view (and remembered,
    for the views met often); several beginnings, the one that explains the stories best is kept"""

    def __init__(self, taught, a, log):
        self.asks, self.answers, self.a = {}, {}, a
        data = []
        for one in taught:
            steps = steps_of(one)
            if steps and len(steps) <= 6:
                data.append((steps, self.asks.setdefault(one[6], len(self.asks)), self.answers.setdefault(one[4], len(self.answers))))
        met = Counter(v for steps, _, _ in data for v, _ in steps)
        self.kinds = {v: i for i, v in enumerate(met)}
        often = [v for v, c in met.items() if c >= a.often]
        vocab = {w for v in met for w in v} | {"<0>", "<1>", "<o>"}
        ids = [([(self.kinds[v], sign) for v, sign in steps], ask, gold) for steps, ask, gold in data]
        offs = walk.steps_of(a.numbers)
        n, choices, best = a.numbers, len(offs), None
        long = [i for i, x in enumerate(data) if len(x[0]) > 1]
        for seed in range(a.seed, a.seed + a.begin):
            torch.manual_seed(seed)
            rng = random.Random(seed)
            net, rate, _ = text.reader(vocab, choices, a)
            net = text.Known(net, often, choices).to(text.DEV)
            says = torch.nn.Parameter((0.1 * torch.randn(len(self.asks), len(offs), len(self.answers))).to(text.DEV))
            opt = torch.optim.Adam(text.rates(net, rate) + [{"params": [says], "lr": 0.1, "base": 0.1}])
            for step in range(a.turns):
                few = long if step < a.turns // 2 and len(long) >= 20 else range(len(data))     # (first the stories of several links:
                batch = [data[i] for i in rng.sample(list(few), min(256, len(few)))]           #  they say how the links add up)
                far = max(len(x[0]) for x in batch)
                net.train()
                b = net([list(v) for steps, _, _ in batch for v, _ in steps]).softmax(-1)
                beliefs = torch.zeros((len(batch), far, choices), device=text.DEV)
                at = 0
                for i, (steps, _, _) in enumerate(batch):
                    beliefs[i, :len(steps)] = b[at:at + len(steps)]
                    at += len(steps)
                sign = torch.tensor([[s for _, s in steps] + [1] * (far - len(steps)) for steps, _, _ in batch], device=text.DEV)
                on = torch.tensor([[True] * len(steps) + [False] * (far - len(steps)) for steps, _, _ in batch], device=text.DEV)
                p = torch.zeros((len(batch),) + (2 * far + 1,) * n, device=text.DEV)
                p[(slice(None),) + (far,) * n] = 1                # the register: how much is believed of everything it may hold
                dims = tuple(range(1, n + 1))
                for i in range(far):
                    add = torch.where(sign[:, i, None] > 0, beliefs[:, i], beliefs[:, i].flip(-1))
                    new = sum(add[(slice(None), j) + (None,) * n] * torch.roll(p, off, dims) for j, off in enumerate(offs))
                    p = torch.where(on[:, i][(slice(None),) + (None,) * n], new, p)
                for d in dims:                                   # of every number: below nought, nought, above
                    p = p.transpose(1, d)
                    p = torch.stack([p[:, :far].sum(1), p[:, far], p[:, far + 1:].sum(1)], 1).transpose(1, d)
                said = (p.reshape(len(batch), -1)[:, :, None] * torch.softmax(says[torch.tensor([q for _, q, _ in batch], device=text.DEV)], -1)).sum(1)
                loss = -torch.log(said[torch.arange(len(batch), device=text.DEV), torch.tensor([g for _, _, g in batch], device=text.DEV)].clamp_min(1e-6)).mean()
                loss = loss + 0.5 * (1 - step / a.turns) * (b * torch.log(b.clamp_min(1e-9))).sum(-1).mean()    # (kept from hardening too early)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            hard = walk.Walk(len(self.kinds), len(self.asks), len(self.answers), 0, a.numbers)
            hard.add, hard.say = self.read(net, list(self.kinds)), says.argmax(-1).tolist()
            first = sum(hard.hard(s, q) == g for s, q, g in ids) / len(ids)
            changed = hard.polish(ids)
            right = sum(hard.hard(s, q) == g for s, q, g in ids) / len(ids)
            log(f"the register, begun from {seed}: made hard, of the stories taught on right {100 * first:.2f}%; every choice tried over ({changed} changed): {100 * right:.2f}%")
            if best is None or right > best[0]:
                best = (right, hard, net)
            if right == 1:
                break
        self.explains, self.hard, self.net = best
        self.weights = sum(p.numel() for p in self.net.parameters())

    @staticmethod
    def read(net, views):
        with torch.no_grad():
            return [k for i in range(0, len(views), 500) for k in net([list(v) for v in views[i:i + 500]]).argmax(-1).tolist()]

    def answer(self, one):
        steps = steps_of(one)
        if not steps or one[6] not in self.asks:
            return None
        new = [v for v, _ in steps if v not in self.kinds]       # (a view not met in teaching is read from its words)
        for v, k in zip(new, self.read(self.net, new)):
            self.kinds[v] = len(self.kinds)
            self.hard.add.append(k)
        word = {i: w for w, i in self.answers.items()}
        return word.get(self.hard.hard([(self.kinds[v], sign) for v, sign in steps], self.asks[one[6]]))


class Tabled:
    """the table taught through readers (text.py --answers): the words a link may be called by are the words of the
    answers and a few spare ones"""

    def __init__(self, taught, a, log):
        text.WORDS = sorted({one[4] for one in taught}) + [f"<spare {i}>" for i in range(a.spare)]
        met = Counter(tuple(text.view(told, p, q, True)) for told, pairs, *_ in taught for p, q in text.together(told))
        self.a = copy.copy(a)
        self.a.known = [v for v, c in met.items() if c >= a.often]
        self.kept, self.answers, *_ = text.minds(taught, self.a, log)
        self.explains = self.kept[0][0]
        self.weights = sum(p.numel() for _, net, *_ in self.kept for p in net.parameters())

    def answer(self, one):
        got, _ = text.voted(self.kept, one, self.a)
        return None if got is None else text.WORDS[got]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepgame", default="", help="the folder TrainVersion of StepGame")
    ap.add_argument("--clutrr", default="", help="a folder with CLUTRR as tables (gen23_train.csv, gen23_test.csv)")
    ap.add_argument("--tests", default="gen23_test.csv", help="--clutrr: the files tried on, with commas between")
    ap.add_argument("--stories", default="", help="stories to teach on, a line each (see above)")
    ap.add_argument("--tried", default="", help="stories to try on, a line each; several files with commas between")
    ap.add_argument("--most", type=int, default=0, help="taught on so many stories only, drawn at random")
    ap.add_argument("--sample", type=int, default=4000, help="the two machines are compared on so many stories (four fifths taught on, a fifth kept aside)")
    ap.add_argument("--often", type=int, default=20, help="a view met so many times in teaching is remembered as it is")
    ap.add_argument("--numbers", type=int, default=2, help="the register: how many numbers it holds")
    ap.add_argument("--begin", type=int, default=6, help="the register: so many beginnings")
    ap.add_argument("--turns", type=int, default=300, help="the register: steps of teaching")
    ap.add_argument("--spare", type=int, default=4, help="the table: so many names beyond the words of the answers")
    ap.add_argument("--readers", type=int, default=5, help="the table: so many readers, each with a table of its own")
    ap.add_argument("--steps", type=int, default=2500, help="the table: steps of teaching")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--encoder", default="", help="a network taught beforehand on plain text in the reader's place (a video card)")
    ap.add_argument("--only", default="", help="register or table: the other machine is not tried")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.batch, a.rate, a.short, a.bare, a.lessons = 192, 0.05, False, False, False
    torch.set_num_threads(a.threads)
    rng, t0 = random.Random(a.seed), time.time()

    def log(line):
        print(f"[{time.time() - t0:5.0f} s] {line}", flush=True)

    if a.stepgame:
        taught, tried = stepgame(a.stepgame)
    elif a.clutrr:
        taught, tried = clutrr(a.clutrr, tests=a.tests.split(","))
    else:
        taught, tried = plain(a.stories), {os.path.basename(name): plain(name) for name in a.tried.split(",")}
    taught = rng.sample(taught, a.most) if a.most else taught
    log(f"{len(taught)} stories to teach on, {sum(map(len, tried.values()))} to try on; answers: {' '.join(sorted({one[4] for one in taught}))}")

    machines, kept = {"register": Register, "table": Tabled}, a.only
    if not kept:                                                 # which machine: both taught on four fifths of a sample, the fifth kept aside tells
        some = rng.sample(taught, min(len(taught), a.sample))
        aside, most = some[:len(some) // 5], some[len(some) // 5:]
        quick = copy.copy(a)
        quick.begin, quick.readers, quick.steps = 2, 1, 800
        share = {}
        for name, machine in machines.items():
            model = machine(most, quick, lambda line: None)
            share[name] = 100 * sum(model.answer(one) == one[4] for one in aside) / len(aside)
            log(f"the {name}: of the {len(aside)} stories kept aside it explains {share[name]:.1f}%")
        kept = max(share, key=share.get)
    log(f"kept: the {kept}; now taught on all the stories")
    model = machines[kept](taught, a, log)
    log(f"the {kept} explains {100 * model.explains:.2f}% of the stories taught on; {model.weights} weights in the reading")

    print("\nthe stories tried, answered rightly, %:")
    for name, some in tried.items():
        by = {}
        for one in some:
            by.setdefault(one[5], []).append(model.answer(one) == one[4])
        print(f"  {name:24s}" + "".join(f"  {k}: {100 * sum(v) / len(v):5.1f}" for k, v in sorted(by.items())) + f"   all: {100 * sum(map(sum, by.values())) / len(some):5.1f}  ({len(some)} stories)", flush=True)
    log("done")


if __name__ == "__main__":
    main()
