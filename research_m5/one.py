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
  reading without answers
             the register reads the stories it is to be asked about before it is asked: where a sentence of a kind
             not met in teaching links two things and another way of sentences understood leads between them,
             the register must come to the same by both, and that says what the sentence means
  learning   by the answers alone: the choices are soft and a gradient goes through the machine; then they are made
             hard and counted exactly on the stories taught on (text.py, walk.py). Which machine suits the stories
             is told by the stories themselves: both are taught on those with short ways and tried on those with
             longer ones, and the one that carries further is kept

    python one.py --stepgame .../StepGame/Dataset/TrainVersion
    python one.py --clutrr DIR            (gen23_train.csv and gen23_test.csv, as for text.py)
    python one.py --babi .../en/qa17_positional-reasoning_train.txt
    python one.py --stories teach.jsonl --tried try.jsonl
                                          (a line: {"story": "... [Ann] ... [Bob] ...", "asked": ["Ann", "Bob"],
                                                    "question": "", "answer": "sister", "k": 3})
"""
import argparse
import ast
import copy
import csv
import glob
import json
import os
import random
import re
import signal
import tempfile
import time
from collections import Counter

import torch

import recall
import rows
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


def babi(teach, tests):
    """a task of bAbI about two things (17, 18): the things are what stands after "the" at the head of a sentence,
    up to the verb; every question of a story is a story of its own"""
    def lines(path):
        told = []
        for line in open(path, encoding="utf-8"):
            number, rest = line.rstrip("\n").split(" ", 1)
            if number == "1":
                told = []
            if "\t" in rest:
                question, answer = rest.split("\t")[:2]
                yield list(told), question, answer
            else:
                told.append(rest)
    things = sorted({m.group(1).lower() for path in [teach] + list(tests) for told, _, _ in lines(path) for s in told
                     for m in [re.match(r"The (.+?) (?:is|fits|are|fit)\b", s)] if m}, key=len, reverse=True)
    find = re.compile(r"\b[Tt]he (" + "|".join(map(re.escape, things)) + r")\b") if things else None
    mark = lambda text: find.sub(lambda m: "[" + m.group(1).lower() + "]", text) if find else text

    def read(path):
        out = []
        for told, question, answer in lines(path):
            marked = mark(question)
            asked = (re.findall(r"\[([^\]]+)\]", marked) + ["", ""])[:2]
            out.append(story([mark(x) for x in told], asked, re.sub(r"\[[^\]]+\]", "<>", marked).lower(), answer, len(told)))
        return out
    return read(teach), {os.path.basename(name): read(name) for name in tests}


def plain(path):
    return [story(x["story"], x["asked"], x.get("question", ""), x["answer"], x.get("k", 0)) for x in map(json.loads, open(path))]


def steps_of(one):
    """the shortest way between the two asked about, every link as (the view of the pair in the order the sentence
    names them, +1 if gone that way, -1 if against it) -> the way or None"""
    told, pairs, _, (x, y), *_ = one
    if x is None or y is None:
        return None
    together = text.together(told)                               # (by names that stand in one sentence; only if there is no such way,
    way = text.surest(sorted(together), lambda p, q: 1, x, y) or text.surest(pairs, lambda p, q: 1 if (p, q) in together else 4, x, y)    #  by sentences next to each other too)
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

    def close(self, stories, log):
        """stories read WITHOUT their answers. Where a sentence of a kind not met in teaching links two things and
        another way of sentences already understood leads between them, the register must come to the same by
        both: that says what the sentence adds. What is settled so helps to settle the rest. A kind read from its
        words very surely is taken at its word to begin with, and is set right if the ways say otherwise"""
        offs, links, new = self.hard.offs, [], set()
        for told, *_ in stories:
            links.append([])
            for p, q in sorted(text.together(told)):
                if p < q:
                    first = next(w for s in told if p in s and q in s for w in s if w in (p, q))
                    a, b = (p, q) if first == p else (q, p)
                    links[-1].append((tuple(text.view(told, a, b, True)), a, b))
                    if links[-1][-1][0] not in self.kinds:
                        new.add(links[-1][-1][0])
        if not new:
            return
        views = sorted(new)
        with torch.no_grad():
            said = torch.cat([self.net([list(v) for v in views[i:i + 500]]).softmax(-1).cpu() for i in range(0, len(views), 500)])
        words = {v: int(said[i].argmax()) for i, v in enumerate(views)}
        sure = {v: words[v] for i, v in enumerate(views) if float(said[i].max()) >= 0.9}
        settled = {}
        for turn in range(8):
            def known(v):
                return self.hard.add[self.kinds[v]] if v in self.kinds else settled.get(v, sure.get(v))
            votes = {}
            for story in links:
                for i, (v, a, b) in enumerate(story):
                    if v in new:                                 # another way from a to b, by sentences understood, none of this kind
                        at, edge = {a: (0,) * len(offs[0])}, [a]
                        while edge and b not in at:
                            x = edge.pop(0)
                            for j, (w, c, d) in enumerate(story):
                                k = known(w) if j != i and w != v and x in (c, d) else None
                                y = d if x == c else c
                                if k is not None and y not in at:
                                    at[y] = tuple(t + (o if x == c else -o) for t, o in zip(at[x], offs[k]))
                                    edge.append(y)
                        if b in at and at[b] in offs:
                            votes.setdefault(v, Counter())[offs.index(at[b])] += 1
            changed = 0
            for v, c in votes.items():
                top, times = c.most_common(1)[0]
                if times >= 2 and times >= 0.8 * sum(c.values()) and settled.get(v) != top:
                    settled[v], changed = top, changed + 1
            if not changed:
                break
        against = sum(settled[v] != words[v] for v in settled)
        log(f"read without answers: {len(new)} kinds of sentences not met in teaching; {len(settled)} settled by the register having to come to the same "
            f"by two ways ({against} of them against what the words said), {len(new) - len(settled)} left to the words")
        for v in views:
            self.kinds[v] = len(self.kinds)
            self.hard.add.append(settled.get(v, words[v]))

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


def as_told(stories, path):
    """stories of the common form written as stories told in time (the form of bAbI: the sentences, then the
    question with its answer), for the machine of recalling -> the path"""
    name = lambda i: "p" + "".join(chr(97 + int(d)) for d in str(i))
    with open(path, "w", encoding="utf-8") as out:
        for told, _, _, (x, y), answer, _, question in stories:
            for n, sentence in enumerate(told):
                out.write(f"{n + 1} " + " ".join(name(w) if w.__class__ is int else w for w in sentence if w.__class__ is int or w not in ".,!?") + "\n")
            asked = iter([name(x) if x is not None else "nobody", name(y) if y is not None else "nobody"])
            words = re.sub("<>", lambda m: next(asked, "nobody"), question) if "<>" in question else f"what is {next(asked)} to {next(asked)}"
            out.write(f"{len(told) + 1} {words}\t{answer}\t1\n")
    return path


class Recalled:
    """the third machine, joined as it is (recall.py): a story is kept as it was told, statement after statement,
    and a question is answered by recalling - the latest value of a thing, where a chain of values ends, who has a
    value, how many. It is for stories told in time with questions between (the form of bAbI); it finds the things
    by itself (the words that take turns in one place of a sentence) and learns from the first twenty stories and
    up to twenty more that it asks for. A hundred stories more are kept aside to tell what it explains. Its time is
    bounded: a task it does not learn in time is not its task"""

    def __init__(self, path, a, log):
        raw, self.m, self.aside, self.used = rows.stories(path, 140), None, 0.0, 0

        def late(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, late)
        signal.alarm(a.patience)
        try:
            tried, plain = [], None                              # (the stories as they are, then with one guess about their words: recall.py)
            for deep in (False, True):
                for reader in rows.guesses(raw[:20]):
                    for split in (False, True):
                        m = recall.Learner(reader.rows(raw[:20]), split, deep=deep)
                        right, total = m.explained()
                        plain = plain or len(m.readings)
                        tried.append((right - 0.5 * len(m.readings) - m.cost() - 3 * bool(reader.guessed), right == total, reader, split))
                        if right == total or not m.task.split:
                            break
                    if tried[-1][1] and len(m.readings) <= plain:
                        break
                best = max(tried, key=lambda t: t[0])
                if best[1]:
                    break
            self.reader = best[2]
            self.m, asked, _ = recall.study(self.reader, raw[:40], 20, 20, best[3])
            right, total, _, _ = self.m.test(self.reader.rows(raw[40:140]))
            self.aside, self.used = 100 * right / max(1, total), 20 + asked
        except TimeoutError:
            self.m = None
        finally:
            signal.alarm(0)

    def tried(self, path):
        right, total, _, _ = self.m.test(self.reader.rows(rows.stories(path)))
        return right, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepgame", default="", help="the folder TrainVersion of StepGame")
    ap.add_argument("--clutrr", default="", help="a folder with CLUTRR as tables (gen23_train.csv, gen23_test.csv)")
    ap.add_argument("--tests", default="gen23_test.csv", help="--clutrr: the files tried on, with commas between")
    ap.add_argument("--babi", default="", help="a task of bAbI: its file to teach on (..._train.txt), tried on the file ..._test.txt beside it; or a folder of such tasks")
    ap.add_argument("--tasks", default="", help="--babi with a folder: the numbers of the tasks, with commas between (all by default)")
    ap.add_argument("--patience", type=int, default=300, help="--babi: so many seconds are given to the machine of recalling")
    ap.add_argument("--stories", default="", help="stories to teach on, a line each (see above)")
    ap.add_argument("--tried", default="", help="stories to try on, a line each; several files with commas between")
    ap.add_argument("--most", type=int, default=0, help="taught on so many stories only, drawn at random")
    ap.add_argument("--sample", type=int, default=4000, help="the two machines are compared on so many stories")
    ap.add_argument("--few", type=int, default=0, help="of every set tried so many stories only, drawn at random (0: all)")
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
    ap.add_argument("--unread", action="store_true", help="the stories tried are not read without their answers beforehand")
    ap.add_argument("--tests-in", default="", help="--stepgame: the stories tried are taken from this folder (the stories taught on stay)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.batch, a.rate, a.short, a.bare, a.lessons = 192, 0.05, False, False, False
    torch.set_num_threads(a.threads)
    if os.path.isdir(a.babi):                                    # every task of the folder in its turn
        wanted = {int(k) for k in a.tasks.split(",") if k}
        for path in sorted(glob.glob(os.path.join(a.babi, "qa*_train.txt")), key=lambda f: int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1))):
            if not wanted or int(re.match(r"qa(\d+)_", os.path.basename(path)).group(1)) in wanted:
                print(f"\n=== {os.path.basename(path)[:-10]}", flush=True)
                one = copy.copy(a)
                one.babi = path
                solve(one)
        return
    solve(a)


def solve(a):
    rng, t0 = random.Random(a.seed), time.time()

    def log(line):
        print(f"[{time.time() - t0:5.0f} s] {line}", flush=True)

    if a.stepgame:
        taught, tried = stepgame(a.stepgame)
        tried = stepgame(a.tests_in)[1] if a.tests_in else tried
    elif a.clutrr:
        taught, tried = clutrr(a.clutrr, tests=a.tests.split(","))
    elif a.babi:
        taught, tried = babi(a.babi, [a.babi.replace("_train", "_test")])
    else:
        taught, tried = plain(a.stories), {os.path.basename(name): plain(name) for name in a.tried.split(",")}
    taught = rng.sample(taught, a.most) if 0 < a.most < len(taught) else taught
    log(f"{len(taught)} stories to teach on, {sum(map(len, tried.values()))} to try on; answers: {' '.join(sorted({one[4] for one in taught}))}")

    machines, kept, share, third = {"register": Register, "table": Tabled}, a.only, {}, None
    two = sum(None not in one[3] for one in taught) >= 0.9 * len(taught)         # (are the questions about two things named in the story?)
    if not kept:                                                 # the machine of recalling: it reads the stories as told in time (for the sets
        folder = tempfile.mkdtemp()                              # that are not in that form, they are written out so: sentences, then the question)
        third = Recalled(a.babi or as_told(taught[:140], os.path.join(folder, "taught.txt")), a, log)
        share["recalling"] = third.aside
        log(f"recalling: " + (f"learnt from {third.used} stories; of the questions of a hundred stories more, kept aside, it explains {third.aside:.1f}%" if third.m
                              else f"not learnt in {a.patience} seconds"))
    if not two and not kept:
        log("the questions are not about two things named in the story: the register and the table are not tried")
    if not kept and two:                                         # which machine: both are taught on the stories of a sample with the shorter ways
        some = rng.sample(taught, min(len(taught), a.sample))    # and tried on those with the longer ones - the machine is kept that carries
        far = [len(steps_of(one) or []) for one in some]         # what it learnt further. (If all the ways are of one length: a fifth kept aside)
        short = max(2, sorted(far)[len(far) // 2])
        aside, most = [one for one, n in zip(some, far) if n > short], [one for one, n in zip(some, far) if 0 < n <= short]
        if len(aside) < 50:
            aside, most = some[:len(some) // 5], some[len(some) // 5:]
            log(f"the two machines are compared on a fifth of {len(some)} stories kept aside")
        else:
            log(f"the two machines are taught on {len(most)} stories with ways of up to {short} links and compared on {len(aside)} stories with longer ways")
        quick = copy.copy(a)
        quick.begin, quick.readers, quick.steps = 2, 1, 800
        for name, machine in machines.items():
            model = machine(most, quick, lambda line: None)
            share[name] = 100 * sum(model.answer(one) == one[4] for one in aside) / len(aside)
            log(f"the {name}: of the {len(aside)} stories kept aside it explains {share[name]:.1f}%")
    kept = kept or max(share, key=share.get)
    if kept == "recalling":
        if not third.m:
            return log("no machine learnt the task")
        for name, some in ([("", None)] if a.babi else tried.items()):
            right, total = third.tried(a.babi.replace("_train", "_test") if a.babi else as_told(some, os.path.join(folder, "tried.txt")))
            print(f"kept: recalling (learnt from {third.used} stories). The questions of the test {name} answered rightly: {100 * right / max(1, total):.1f}% ({total})", flush=True)
        return log("done")
    log(f"kept: the {kept}; now taught on all the stories")
    model = machines[kept](taught, a, log)
    log(f"the {kept} explains {100 * model.explains:.2f}% of the stories taught on; {model.weights} weights in the reading")

    def shown(how):
        print(f"\nthe stories tried, answered rightly, %{how}:")
        for name, some in tried.items():
            by = {}
            for one in some:
                by.setdefault(one[5], []).append(model.answer(one) == one[4])
            print(f"  {name:24s}" + "".join(f"  {k}: {100 * sum(v) / len(v):5.1f}" for k, v in sorted(by.items())) + f"   all: {100 * sum(map(sum, by.values())) / len(some):5.1f}  ({len(some)} stories)", flush=True)

    tried = {name: rng.sample(some, a.few) if 0 < a.few < len(some) else some for name, some in tried.items()}
    if kept == "register" and not a.unread:                      # (the register can read the stories it will be asked about, without their answers)
        fresh = copy.deepcopy((model.kinds, model.hard.add))
        new = sum(v not in model.kinds for some in tried.values() for one in some for v, _ in steps_of(one) or [])
        if new:
            shown(" (sentences not met in teaching read from their words alone)")
            model.kinds, model.hard.add = fresh
            model.close([one for some in tried.values() for one in some], log)
            shown(" (after the stories tried were read without their answers)")
            return log("done")
    shown("")
    log("done")


if __name__ == "__main__":
    main()
