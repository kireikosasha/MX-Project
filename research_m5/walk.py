#!/usr/bin/env python3
"""
A walk with a register over links read from sentences: the benchmark StepGame (Shi, Zhang, Lipani, 2022).

A story of StepGame says where things stand against each other ("Y is at a 45 degree angle to I, in the lower
righthand corner.") in some three hundred ways; the question asks how one thing stands against another, which
takes up to ten such sentences one after another, among sentences that do not matter. The models of the paper are
taught on 50 000 stories of one to five steps and tried on ten thousand stories of every number of steps up to ten.

Here nothing is said of space. A sentence with two names is a link between two things; of every kind of sentence
(the sentence with the names taken out) it is learnt what it adds to a register - a few whole numbers (two by
default), each of them -1, 0 or 1 more - softly. The question is answered by a walk: from one thing to the other
along the links (the shortest way, found exactly), the register taking every link in, against the link if it is
gone backwards. What is said is learnt too: a table from the signs of the numbers of the register to the words of
the answers. The state of the walk is how much is believed of everything the register may hold; the gradient of
the answers goes through the walk. Then the choices are made hard, the walk is run on whole numbers, and every
choice is tried over against the stories taught on (exactly: one choice changed, the stories counted).

    python walk.py --data .../StepGame/Dataset/TrainVersion
    python walk.py --data ... --stories 300 --least 2         # taught on 300 stories, none answered by one sentence
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the folder TrainVersion of StepGame")
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
    keys = [i for i, x in enumerate(read(os.path.join(a.data, "train.json"), {}, {}, {}, True)) if x[0] is not None and len(x[0]) >= a.least]
    keys = set(rng.sample(keys, a.stories) if a.stories else keys)
    taught = [x for i, x in enumerate(read(os.path.join(a.data, "train.json"), kinds, asks, answers, True, keys)) if x[0] is not None]
    print(f"taught on {len(taught)} stories (the answer takes {a.least} sentences and more): {len(kinds)} kinds of sentences, {len(asks)} of questions, "
          f"{len(answers)} answers", flush=True)

    best, long = None, [x for x in taught if len(x[0]) > 1]
    for seed in range(a.seed, a.seed + a.begin):                 # (begun anew several times: about one beginning in four ends in choices
        model = Walk(len(kinds), len(asks), len(answers), seed, a.numbers)       # that do not add up; the exact count on the stories taught on tells)
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
        print(f"begun from {seed}: loss {loss.item():.3f}; made hard, of the stories taught on right {100 * first:.2f}%; "
              f"every choice tried over ({changed} changed): {100 * right:.2f}%", flush=True)
        if best is None or right > best[0]:
            best = (right, model)
        if right == 1:
            break
    model = best[1]
    print(f"kept: the beginning that explains {100 * best[0]:.2f}% of the stories taught on; {sum(p.numel() for p in model.parameters())} numbers learnt, "
          f"{time.time() - t0:.0f} s with all the beginnings", flush=True)

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
