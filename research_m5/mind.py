#!/usr/bin/env python3
"""
A network that answers by itself: no ground at work, no words, no search.

guess.py names a reading and lets the ground execute it. Here the same dreams (relive.py) teach a network to do the
whole thing in its weights: it reads the stories of a task from left to right - which kinds of sentences there are,
which things stand in them, what was answered so far - and at every question says the answer. Nothing tells it what
a kind of sentence means: the names of the kinds and of the things are dealt anew in every task, so the meaning can
only be taken from the stories before, as the learner takes it. What it has to grow is the ground itself: keeping
who is where, following a chain, counting.

Builds of the same network are compared. Plain: layers piled one upon another. A loop: a few layers gone through
several times with the same weights (as much computing, fewer weights). And with a memory of what was said, kept by
name (--names): every statement of the story stays as a record - its kind and the things in it - and at any place
the network may recall the latest records that name a thing. Which thing, of which kind of sentence, and what to do
with the record recalled, it learns; that the name is found exactly, and the latest mention first, is built in. An
answer is then given the same way: by pointing at a name.

    python mind.py --lived lived.pkl --hold H,I,J,K --minutes 100 --out mind.pt         # taught on dreams, tried on tasks it has not seen
    python mind.py --lived lived.pkl --hold H,I,J,K --net mind.pt                        # only tried
"""
import argparse
import copy
import math
import os
import pickle
import random
import time

import torch
from torch import nn

import guess
import records
import relive
from guess import ANS, EACH, KINDS, N0, NUMS, OTHER, PAD, PLAIN, Q0, S0, STORY, T0, VOCAB, order

EOA = VOCAB                                                     # the answer is said in full
SIZE = VOCAB + 1
RECORD = 5                                                      # a record: the kind of sentence and up to four things of it
RECENT = 5                                                      # how many of the latest records that name a thing may be recalled
RECALLED = 40                                                   # how many mentions in all may be recalled at one place
STALE = 3                                                       # how late a mention is, told apart up to this (0: the latest)
COUNTS = 8                                                      # a number of records recalled is told apart up to this


def dealt(episodes):
    """the names of a task dealt once, to be the same whatever stories of it are taken: its kinds of sentences in the
    order they come in all its stories, and the codes its things have been given so far"""
    return order(episodes) + ({}, {})


def tokens(episodes, rng=None, most=1280, deal=None):
    """stories -> the tokens; for every place what is to be said next there (-1: nothing is asked of it); the
    questions of the last story, each as the places its answer is said at; and where every place stands: its story,
    its row, whether it is a thing of a statement, the place its row begins at, and the record of its row (the kind
    of sentence and its things, for a statement). None if the task has more kinds of sentences than the network has
    names for. rng: the names are dealt anew (in teaching). The words of an answer are said in the order of their
    tokens: a list has no order of its own. deal: the names as they were dealt for this task before (dealt()), kept
    the same from call to call. Of every place it is also said what it may recall: the places of the
    latest mentions of things in the statements of its story before its row, how late each mention is, and how late
    the mention of every thing of its record is by now (a record all of whose things are the latest still holds)"""
    said, asked, code, kinds = deal or (order(episodes) + ({}, {}))
    if len(said) > guess.STATED or len(asked) > guess.ASKED:
        return None
    sid, qid = list(range(guess.STATED)), list(range(guess.ASKED))
    kid, eid = list(range(KINDS)), {}
    if rng:
        rng.shuffle(sid), rng.shuffle(qid), rng.shuffle(kid)
    parts = []

    def thing(x):
        if x not in code:
            k = kinds.setdefault(x[0], kid[len(kinds) % KINDS])
            if k not in eid:
                eid[k] = list(range(EACH))
                if rng:
                    rng.shuffle(eid[k])
            code[x] = T0 + k * EACH + eid[k][sum(1 for y in code if kinds[y[0]] == k) % EACH]
        return code[x]

    for ep in episodes:
        out, aim, spans, rows = [STORY], [-1], [], [(0, 0, 0, (0,) * RECORD)]      # of a place: its row in the story, a thing of a
        for n, (op, quest, things, numbers, ks, gold) in enumerate(ep):            # statement?, its slot, the record of its row
            f = (op, quest, ks)
            head = Q0 + qid[asked.index(f)] if quest else S0 + sid[said.index(f)]
            ts, ns = iter(things), iter(numbers)
            slots = [N0 + min(NUMS - 1, max(0, next(ns))) if k == "#" else thing(next(ts)) for k in ks]
            record = tuple(([head] + slots + [0] * RECORD)[:RECORD])
            out += [head] + slots
            rows += [(n + 1, 0, 0, record)] + [(n + 1, int(not quest), min(i, RECORD - 2), record) for i in range(len(slots))]
            aim += [-1] * (len(out) - len(aim))
            if quest:
                told = sorted({PLAIN[x] if x in PLAIN else N0 + min(NUMS - 1, int(x)) if x.isdigit() else thing(x) if x[:1].isalpha() and x[1:].isdigit() else OTHER
                               for x in gold.split(",")}) + [EOA]
                spans.append(list(range(len(out), len(out) + len(told))))
                out += [ANS] + told[:-1]
                aim += told
                rows += [(n + 1, 0, 0, record)] * len(told)
        parts.append((out, aim, spans, rows))
    while len(parts) > 1 and sum(len(p[0]) for p in parts) > most:      # too long: the first stories are left out
        parts.pop(0)
    if sum(len(p[0]) for p in parts) > most:
        return None
    start = sum(len(p[0]) for p in parts[:-1])
    where, at = [], 0
    for n, (out, _, _, rows) in enumerate(parts):
        kept, mine, row_now = {}, (), -1                         # of every thing: the places of its latest mentions in statements
        rows_at = {at + i: row for i, (row, stated, _, _) in enumerate(rows) if stated}       # of a mention: its row; of a row: where its
        slots_of = {}                                            # things stand (-1: no such slot)
        for i, (row, stated, slot, _) in enumerate(rows):
            if stated:
                slots_of.setdefault(row, [-1] * (RECORD - 1))[slot] = at + i
        for i, (row, stated, slot, record) in enumerate(rows):
            if row != row_now:                                   # a new row: it may recall what the rows before it said
                row_now = row
                late = sorted(((p, r) for ps in kept.values() for r, p in enumerate(ps)), reverse=True)[:RECALLED]
                how = {p: r for ps in kept.values() for r, p in enumerate(ps)}      # of a mention: how late it is by now
                still = tuple(min(STALE, how.get(q, STALE)) if q >= 0 else STALE + 1 for p, _ in late for q in slots_of[rows_at[p]])     # of the things of its record
                still += (STALE + 1,) * ((RECORD - 1) * (RECALLED - len(late)))
                mine = tuple(p for p, _ in late) + (0,) * (RECALLED - len(late)) + tuple(r for _, r in late) + (-1,) * (RECALLED - len(late)) + still
                fresh = []
            where.append((n, row, stated, slot) + record + mine)
            if stated:
                fresh.append((out[i], at + i))
            if i + 1 == len(rows) or rows[i + 1][0] != row:      # the row is said: its things are remembered
                for t, p in fresh:
                    kept[t] = ([p] + kept.get(t, []))[:RECENT]
        at += len(out)
    return [t for p in parts for t in p[0]], [t for p in parts for t in p[1]], [[start + i for i in span] for span in parts[-1][2]], where


class Recall(nn.Module):
    """what was said, recalled by name. A place asks for a name (a vector to be matched with the fixed mark every
    word has), and may say of which kind of sentence, in which slot, and whether the things of the record must not
    have been spoken of since; it is given the records of the latest statements of its story that name it - the kind
    of sentence and the things of each, as their marks - and how many records answered. The match of names is exact
    by build (the marks are not learnt), the latest mention comes first; the rest is learnt"""

    def __init__(self, width, marks, heads=2):
        super().__init__()
        self.heads = heads
        self.name, self.kind = nn.Linear(width, heads * marks), nn.Linear(width, heads * marks)
        self.slot = nn.Linear(width, heads * (RECORD - 1))
        self.still = nn.Linear(width, heads * (RECORD - 1) * (STALE + 2))
        self.sharp = nn.Parameter(torch.full((heads,), 16.0))
        self.late = nn.Parameter(torch.linspace(2.0, -2.0, RECENT).repeat(heads, 1))        # the later a mention, the likelier at first
        self.none = nn.Parameter(torch.zeros(heads))                                       # (nothing may be recalled at all)
        self.value = nn.Linear(heads * (RECORD * marks + (RECORD - 1) * (STALE + 3) + COUNTS), width)
        self.register_buffer("steps", torch.arange(0.0, COUNTS))                           # (a count is told as how near it is to 0, 1, 2 ...)

    def looks(self, h, said):
        """how much every place looks at each of the mentions it may recall (the rest: at nothing), for every head;
        and how many mentions answer what it asks (as much as nothing does, or more)"""
        named, kinds, slots, rank, _, _, fresh = said
        b, n, _ = h.shape
        ask = nn.functional.normalize(self.name(h).view(b, n, self.heads, -1), dim=3)
        of = self.kind(h).view(b, n, self.heads, -1)
        slot = self.slot(h).view(b, n, self.heads, -1)
        still = self.still(h).view(b, n, self.heads, -1)
        out, many = [], []
        for k in range(self.heads):
            score = self.sharp[k] * torch.einsum("bnc,bnmc->bnm", ask[:, :, k], named) + torch.einsum("bnc,bnmc->bnm", of[:, :, k], kinds) \
                + torch.einsum("bns,bnms->bnm", slot[:, :, k], slots) + torch.einsum("bnf,bnmf->bnm", still[:, :, k], fresh) \
                + self.late[k][rank.clamp(0, RECENT - 1)]
            score = score.masked_fill(rank < 0, -1e9)
            many.append(torch.sigmoid(score - self.none[k]).sum(2))
            out.append(torch.softmax(torch.cat([score, self.none[k].expand(b, n, 1)], 2), 2)[:, :, :-1])
        return out, many

    def forward(self, h, said):
        looks, many = self.looks(h, said)
        seen = [torch.cat([torch.einsum("bnm,bnmr->bnr", look, said[4]), torch.exp(-(count[:, :, None] - self.steps) ** 2)], 2) for look, count in zip(looks, many)]
        return self.value(torch.cat(seen, 2))


class Mind(nn.Module):
    def __init__(self, width=160, layers=4, heads=4, most=1280, loops=1, names=0, marks=32):
        super().__init__()
        self.most, self.loops, self.names = most, loops, names   # loops: the same layers gone through so many times
        self.tok, self.pos = nn.Embedding(SIZE, width, padding_idx=PAD), nn.Embedding(most, width)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, "gelu", batch_first=True, norm_first=True) for _ in range(layers))
        self.norm, self.out = nn.LayerNorm(width), nn.Linear(width, width)
        self.bias = nn.Parameter(torch.zeros(SIZE))
        if names:                                                # names: what was said is kept as records and recalled by name
            mark = nn.functional.normalize(torch.randn(SIZE, marks, generator=torch.Generator().manual_seed(7)), dim=1)
            mark[PAD] = 0
            self.register_buffer("mark", mark)                   # the mark of a word: fixed, never learnt - a name is itself and no other
            self.mine = nn.Linear(RECORD * marks, width)         # a place knows the record of its own row
            self.recall = nn.ModuleList(Recall(width, marks, names) for _ in range(layers))
            self.last, self.which = Recall(width, marks, 1), nn.Linear(width, RECORD)      # an answer may be a word of a record recalled

    def said(self, x, where):
        """what may be recalled at every place: of the mentions listed for it - the mark of the thing named, of the
        kind of its sentence, its slot, how late it is, the record of its row (the marks of its kind and things),
        and the words of that record"""
        b, n = x.shape
        at, rank = where[:, :, 4 + RECORD:4 + RECORD + RECALLED], where[:, :, 4 + RECORD + RECALLED:4 + RECORD + 2 * RECALLED]
        fresh = nn.functional.one_hot(where[:, :, 4 + RECORD + 2 * RECALLED:].view(b, n, RECALLED, RECORD - 1), STALE + 2).float().flatten(3)
        flat = at.reshape(b, -1)
        thing = x.gather(1, flat).view(b, n, RECALLED)
        slot = where[:, :, 3].gather(1, flat).view(b, n, RECALLED)
        rows = where[:, :, 4:4 + RECORD].gather(1, flat[:, :, None].expand(b, n * RECALLED, RECORD)).view(b, n, RECALLED, RECORD)
        slots = nn.functional.one_hot(slot, RECORD - 1).float()
        return self.mark[thing], self.mark[rows[..., 0]], slots, rank, torch.cat([self.mark[rows].flatten(3), slots, fresh], 3), rows, fresh

    def forward(self, x, where=None):
        """-> for every place, how likely every word is to be said next (as logarithms)"""
        b, n = x.shape
        ahead = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)      # a place sees only what stands before it
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))[None]
        said = None
        if self.names:
            h = h + self.mine(self.mark[where[:, :, 4:4 + RECORD]].flatten(2))
            with torch.no_grad():
                said = self.said(x, where)
        for _ in range(self.loops):
            for k, layer in enumerate(self.layers):
                if said:
                    h = h + self.recall[k](h, said)
                h = layer(h, src_mask=ahead)
        h = self.norm(h)
        out = torch.log_softmax(self.out(h) @ self.tok.weight.T + self.bias, 2)     # a word is said by the vector it has on the way in,
        if self.names:                                                               # or it is a word of a record recalled, taken as it is
            look = self.last.looks(h, said)[0][0]
            part = (look[:, :, :, None] * torch.softmax(self.which(h), 2)[:, :, None, :]).reshape(b, n, -1)
            copied = torch.zeros(b, n, SIZE, device=x.device, dtype=part.dtype).scatter_add_(2, said[5].reshape(b, n, -1), part)
            out = torch.log(copied + (1 - look.sum(2, keepdim=True)).clamp(min=0) * out.exp() + 1e-9)
        return out


class Dreams(torch.utils.data.IterableDataset):
    def __init__(self, tasks, a, turn=0):
        self.tasks, self.a, self.turn = tasks, a, turn

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(1000003 * self.a.seed + 1009 * self.turn + (info.id if info else 0))
        for eps, _, _ in relive.dreams(self.tasks, rng, (min(self.a.least, self.a.stories), self.a.stories), self.a.varied):
            got = tokens(eps, rng, self.a.most)
            if got:
                yield got[0], got[1], got[3]


def batch(items, device):
    width = max(len(t) for t, _, _ in items)
    x = torch.zeros(len(items), width, dtype=torch.long)
    y = torch.full((len(items), width), -1, dtype=torch.long)
    w = torch.zeros(len(items), width, 4 + RECORD + (RECORD + 1) * RECALLED, dtype=torch.long)
    w[:, :, 4 + RECORD + RECALLED:4 + RECORD + 2 * RECALLED] = -1
    w[:, :, 4 + RECORD + 2 * RECALLED:] = STALE + 1
    for i, (t, aim, where) in enumerate(items):
        x[i, :len(t)], y[i, :len(t)], w[i, :len(t)] = torch.tensor(t), torch.tensor(aim), torch.tensor(where)
    return x.to(device), y.to(device), w.to(device)


def keep(path, net, opt, a, spent, step, seen, names):
    torch.save({"weights": net.state_dict(), "opt": opt.state_dict(), "size": (a.width, a.layers, a.heads, a.most, a.loops, a.names),
                "spent": spent, "step": step, "seen": seen, "taught on": names}, path + ".part")
    os.replace(path + ".part", path)


def teach(tasks, a, device, log=print):
    """the network taught on dreams of the tasks for so many minutes (a run cut short goes on from what it wrote)"""
    torch.manual_seed(a.seed)
    net = Mind(a.width, a.layers, a.heads, a.most, a.loops, a.names).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    before, step, seen = 0.0, 0, 0
    if a.out and os.path.exists(a.out):
        saved = torch.load(a.out, map_location=device)
        net.load_state_dict(saved["weights"])
        opt.load_state_dict(saved["opt"])
        before, step, seen = saved["spent"], saved["step"], saved["seen"]
        log(f"goes on from {before:.1f} min, step {step}")
    stream = iter(torch.utils.data.DataLoader(Dreams(tasks, a, step), batch_size=a.batch, num_workers=a.workers, collate_fn=list))
    t0, last, avg = time.time(), time.time(), None
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    names = [t["name"] for t in tasks]
    while before + (time.time() - t0) / 60 < a.minutes:
        x, y, w = batch(next(stream), device)
        done = (before + (time.time() - t0) / 60) / a.minutes
        for group in opt.param_groups:
            group["lr"] = a.lr * min(1, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, done))))
        with torch.autocast(device, enabled=amp):
            out = net(x, w).float()
            loss = nn.functional.cross_entropy(out[y >= 0], y[y >= 0])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step, seen = step + 1, seen + x.shape[0]
        hit = (out.argmax(2) == y) | (y < 0)
        late = torch.arange(x.shape[1], device=device)[None] >= (x != PAD).sum(1, keepdim=True) * 2 // 3      # the last third of every row
        now = [loss.item(), hit[y >= 0].float().mean().item(), hit[(y >= 0) & late].float().mean().item()]
        avg = now if avg is None else [0.98 * u + 0.02 * v for u, v in zip(avg, now)]
        if step % a.every == 0:
            log(f"step {step}, {seen} dreams, {before + (time.time() - t0) / 60:.1f} min: loss {avg[0]:.3f}; words of the answers said right {100 * avg[1]:.0f}%, "
                f"in the last third of the stories {100 * avg[2]:.0f}%")
        if a.out and time.time() - last > 60 * a.save:
            keep(a.out, net, opt, a, before + (time.time() - t0) / 60, step, seen, names)
            last = time.time()
    del stream
    if a.out:
        keep(a.out, net, opt, a, a.minutes, step, seen, names)
    return net


def answers(net, shown, later, device, most=40, deal=None):
    """the network answering by itself: the stories shown with their answers, then a later story - and so for every
    later story (at most so many). An answer counts if every word of it is said right -> answers right, all"""
    right = total = 0
    net.eval()
    for ep in later[:most]:
        got = tokens(shown + [ep], None, net.most, deal)
        if not got or not got[2]:
            continue
        with torch.no_grad():
            said = net(torch.tensor([got[0]], device=device), torch.tensor([got[3]], device=device))[0].argmax(1).cpu().tolist()
        for span in got[2]:
            right += all(said[i] == got[1][i] for i in span)
            total += 1
    return right, total


def tuned(net, shown, deal, device, steps, lr=3e-4, seed=0):
    """the network after learning a task from its own stories: so many small steps on them (a dozen of the stories
    at a time, in another order each time; the names stay as dealt, so what a kind of sentence means can settle in
    the weights) -> a copy of the network that has learnt the task"""
    mine = copy.deepcopy(net)
    mine.train()
    opt = torch.optim.AdamW(mine.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    for _ in range(steps):
        items = []
        for _ in range(4):
            got = tokens(rng.sample(shown, min(len(shown), 12)), None, mine.most, deal)
            if got:
                items.append((got[0], got[1], got[3]))
        if not items:
            break
        x, y, w = batch(items, device)
        out = mine(x, w).float()
        loss = nn.functional.cross_entropy(out[y >= 0], y[y >= 0])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mine.parameters(), 1.0)
        opt.step()
    return mine


def trial(net, tasks, device, stories=20, tune=(), log=print):
    """the network on tasks it was not taught on: the first stories of a task with their answers before it, the
    questions of the later stories to answer; the same with fewer stories before it, or none; and (tune: after how
    many steps) after learning the task from its first twenty stories by small steps of its own"""
    rows, before = [], stories - 1                               # (with the story asked about: as many as it was taught on)
    for t in tasks:
        deal = dealt(t["episodes"])
        eps, later = t["episodes"][:20], t["episodes"][20:]
        got = [answers(net, eps[:n], later, device, deal=deal) for n in (before, 5, 0)]
        t0 = time.time()
        for steps in tune:
            got.append(answers(tuned(net, eps, deal, device, steps), eps[:before], later, device, deal=deal))
        rows.append((t["name"], t["full"], got))
        log(f"{t['name']:5} {'learnt' if t['full'] else 'not learnt':10} answers right with {before} stories before: {got[0][0]}/{got[0][1]}; with 5: {got[1][0]}/{got[1][1]}; with none: {got[2][0]}/{got[2][1]}"
            + "".join(f"; after {steps} steps on its 20 stories: {g[0]}/{g[1]}" for steps, g in zip(tune, got[3:])) + (f" ({time.time() - t0:.0f} s)" if tune else ""))
    for label, pick in (("tasks the learner had learnt in full", [r for r in rows if r[1]]), ("tasks it had not", [r for r in rows if not r[1]]), ("all the tasks left out", rows)):
        if pick:
            share = [100 * sum(r[2][i][0] for r in pick) / max(1, sum(r[2][i][1] for r in pick)) for i in range(3 + len(tune))]
            each = [[100 * r[2][i][0] / max(1, r[2][i][1]) for r in pick] for i in range(3 + len(tune))]
            log(f"{label}: {len(pick)}. The network answering by itself - with {before} stories before {share[0]:.1f}%, with 5 {share[1]:.1f}%, with none {share[2]:.1f}%; "
                f"tasks at 95% and more {sum(e >= 95 for e in each[0])}, at 80% and more {sum(e >= 80 for e in each[0])}"
                + "".join(f"; after {steps} steps on the 20 stories of a task {share[3 + i]:.1f}%, tasks at 95% and more {sum(e >= 95 for e in each[3 + i])}, at 80% and more {sum(e >= 80 for e in each[3 + i])}"
                          for i, steps in enumerate(tune)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lived", required=True, help="the tasks as they were understood (made by relive.py)")
    ap.add_argument("--hold", default="", help="sets left out of teaching and tried on: H,I,J,K")
    ap.add_argument("--only", default="", help="teach on these tasks alone (a check): en1,H1")
    ap.add_argument("--net", default="", help="a network already taught")
    ap.add_argument("--out", default="")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--stories", type=int, default=20)
    ap.add_argument("--least", type=int, default=8, help="a dream has from so many stories to --stories")
    ap.add_argument("--varied", type=float, default=0.5, help="how many of the dreams are tasks that never were")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--loops", type=int, default=1, help="the layers are gone through so many times (the same weights): a loop instead of a taller pile")
    ap.add_argument("--names", type=int, default=0, help="what was said is kept as records and recalled by name: so many recalls at every layer (0: none, the plain network)")
    ap.add_argument("--most", type=int, default=1280, help="the longest row of tokens")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--few", type=int, default=0, help="try on only so many of the tasks left out (a short check that everything runs)")
    ap.add_argument("--tune", default="", help="in the trial the network also learns every task from its 20 stories by so many small steps: 50,200")
    ap.add_argument("--save", type=float, default=10)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = pickle.load(open(a.lived, "rb"))
    hold = tuple(x for x in a.hold.split(",") if x)
    only = tuple(x for x in a.only.split(",") if x)
    apart = [t for t in tasks if t["name"].rstrip("0123456789") in hold]
    mine = [t for t in tasks if t["full"] and t not in apart and (not only or t["name"] in only)]
    print(f"{len(mine)} tasks to dream of, {len(apart)} left out to try on; {device}", flush=True)
    if a.net:
        saved = torch.load(a.net, map_location=device)
        net = Mind(*saved["size"]).to(device)
        net.load_state_dict(saved["weights"])
    else:
        net = teach(mine, a, device, lambda s: print(s, flush=True))
    print(f"{sum(p.numel() for p in net.parameters())} weights", flush=True)
    tune = tuple(int(x) for x in a.tune.split(",") if x)
    if only and not a.net:                                       # (a check on the tasks taught: their own later stories)
        trial(net, [t for t in tasks if t["name"] in only], device, a.stories, tune, lambda s: print(s, flush=True))
    elif apart:
        trial(net, apart[:a.few or None], device, a.stories, tune, lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
