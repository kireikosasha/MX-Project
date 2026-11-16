#!/usr/bin/env python3
"""
Naming the route: a small network that reads the words of a task and names what recall.py would have to find.

recall.py finds by trying and checking which memory every kind of statement writes and what every kind of question
reads. Here a network is taught on what that search has already found in other tasks (its hard readings, written out
by recall.py --dump): it reads the words of the kinds of sentences of a task - all of them at once, so that the parts
it names fit together - and names, for every kind of statement, the part it plays in each of the three memories,
and for every kind of question, the way of looking and what is asked. Nothing is answered in the weights: what is
named is a reading in the language of recall.py, and recall.py checks it exactly on the stories of the task. What the
check does not confirm is left to the search.

    python recall.py --data new/A --dump found.jsonl                      # what the search has found
    python route.py --found found.jsonl --hold K,L,M2 --out route.pt      # the network taught on it
    python route.py --net route.pt --data new/M2 --stories 3 --text 200   # a new task from three stories
"""
import argparse
import ast
import glob
import json
import os
import re
import time
from collections import Counter

import torch

import recall
import rows
import soft

SORTS = ("value", "link", "amount")


def tup(x):
    return tuple(tup(y) for y in x) if x.__class__ is list else x


def text(template, label, quest):
    return soft.words(template, quest) + ([label] if label else [])


def lesson(found, reworded=None):
    """what the search found in a task -> its kinds of statements (the sentence, its word of a kind, the part it plays
    in every memory: the commonest among its kinds of questions), its kinds of questions (the sentence, the word,
    what it asks), and for a sentence the other wordings of it (reworded: the same with other words, by someone
    who was shown the sentences only)"""
    parts = {}
    for kind in found["kinds"]:
        for template, label, role in kind["parts"]:
            parts.setdefault((template, label), {}).setdefault(kind["sort"], Counter())[repr(tup(role))] += 1
    said = [(t, l, {s: c.most_common(1)[0][0] for s, c in by.items()}) for (t, l), by in parts.items()]
    asked = [(k["question"], k["label"], repr((k["sort"], tup(k["ask"])))) for k in found["kinds"] if k["right"] == k["total"]]
    other = {}
    for group in ("statements", "questions"):
        for template, ways in ((reworded or {}).get(group) or {}).items():
            good = [w.lower().strip() for w in ways if w.__class__ is str and w.count("*") == template.count("*") and w.count("#") == template.count("#")
                    and re.fullmatch(r"[a-z*# ]+", w.lower().strip())]
            other[(group == "questions", template)] = sorted(set(good) - {template})
    return said, asked, other


def worded(lesson_, rng=None):
    """the texts of a lesson: as the sentences were, or (rng) each in one of its wordings drawn at random"""
    said, asked, other = lesson_

    def pick(quest, template):
        ways = [template] + other.get((quest, template), []) if rng else [template]
        return ways[int(torch.randint(len(ways), (), generator=rng))] if rng else template
    return [text(pick(False, t), l, False) for t, l, _ in said], [text(pick(True, t), l, True) for t, l, _ in asked]


class Namer(torch.nn.Module):
    def __init__(self, vocab, parts, asks, width=64):
        super().__init__()
        self.vocab = {w: i + 1 for i, w in enumerate(sorted(vocab))}
        self.parts = {s: ["None"] + sorted(parts[s]) for s in SORTS}             # of a memory: the parts met in the lessons
        self.asks = sorted(asks)
        self.embed = torch.nn.Embedding(len(self.vocab) + 1, width)
        self.gru = torch.nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.heads = torch.nn.ModuleDict({**{s: torch.nn.Linear(6 * width, len(self.parts[s])) for s in SORTS}, "ask": torch.nn.Linear(6 * width, len(self.asks))})
        self.drop = 0.0

    def read(self, texts):
        ids = [[self.vocab.get(w, 0) if not (self.training and torch.rand(()) < self.drop and not w.startswith("<")) else 0 for w in t] for t in texts]
        most = max(len(x) for x in ids)
        x = self.embed(torch.tensor([x + [0] * (most - len(x)) for x in ids]))
        out, _ = self.gru(torch.nn.utils.rnn.pack_padded_sequence(x, [len(i) for i in ids], batch_first=True, enforce_sorted=False))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = torch.tensor([[1.0] * len(i) + [0.0] * (most - len(i)) for i in ids])[:, :, None]
        return (out * mask).sum(1) / mask.sum(1)

    def forward(self, said, asked):
        """the texts of the kinds of statements and of questions of one task -> of every statement the scores of its
        part in each memory, of every question the scores of what it asks; each read with all the others in view"""
        h = self.read(said + asked)
        hs, hq = h[:len(said)], h[len(said):]
        around = torch.cat([hs.mean(0) if len(said) else torch.zeros(h.shape[1]), hq.mean(0) if len(asked) else torch.zeros(h.shape[1])])
        hs, hq = torch.cat([hs, around.expand(len(hs), -1)], 1), torch.cat([hq, around.expand(len(hq), -1)], 1)
        return {s: self.heads[s](hs) for s in SORTS}, self.heads["ask"](hq)


def teach(lessons, steps=400, seed=0, log=print):
    rng = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vocab = {w for said, asked, other in lessons for t, l, _ in said + asked for w in text(t, l, False)} \
        | {w for _, _, other in lessons for ways in other.values() for way in ways for w in way.split() if w not in "*#"} | {"?", "."}
    parts = {s: {by[s] for said, _, _ in lessons for _, _, by in said if s in by} for s in SORTS}
    net = Namer(vocab, parts, {a for _, asked, _ in lessons for _, _, a in asked})
    net.drop = 0.1
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for step in range(steps):
        net.train()
        loss = 0.0
        for one in lessons:
            said, asked, other = one
            if not said or not asked:
                continue
            roles, asks = net(*worded(one, rng if other else None))
            for s in SORTS:
                loss = loss + torch.nn.functional.cross_entropy(roles[s], torch.tensor([net.parts[s].index(by.get(s, "None")) for _, _, by in said]))
            loss = loss + torch.nn.functional.cross_entropy(asks, torch.tensor([net.asks.index(a) for _, _, a in asked]))
        opt.zero_grad()
        (loss / len(lessons)).backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"step {step}: loss {loss.item() / len(lessons):.3f}")
    net.eval()
    return net


def named(net, task, most=3):
    """what the network names for a task -> of every memory the parts of the kinds of statements, of every kind of
    question the few things it most likely asks (way of looking, what is asked)"""
    said = sorted(task.forms.items(), key=lambda x: x[1])
    asked = sorted(task.asked.items(), key=lambda x: x[1])
    with torch.no_grad():
        roles, asks = net([text(f[0], f[3], False) for f, _ in said], [text(f[0], f[3], True) for f, _ in asked])
    parts = {}
    for s in SORTS:
        nt = {n: sum(k != "#" for k in f[2]) for f, n in said}
        parts[s] = {}
        for (f, n), row in zip(said, roles[s]):
            for i in row.argsort(descending=True).tolist():          # the likeliest part this kind of statement has places for
                role = ast.literal_eval(net.parts[s][i])
                if role is None or fits(role, nt[n], sum(k == "#" for k in f[2])):
                    if role is not None:
                        parts[s][n] = role
                    break
    wants = {}
    for (f, q), row in zip(asked, asks):
        nt, out = sum(k != "#" for k in f[2]), []
        for i in row.argsort(descending=True).tolist():
            sort, ask = ast.literal_eval(net.asks[i])
            if fits(ask, nt, sum(k == "#" for k in f[2])):
                out.append((sort, ask))
            if len(out) == most:
                break
        wants[q] = out
    return parts, wants


def fits(x, nt, nn):
    """has a sentence with so many things and numbers the places a part (or a thing asked) speaks of?"""
    if x.__class__ is tuple:
        if x and x[0] in ("when", "asof", "until") and x[1].__class__ is int:
            return x[1] < nn and fits(x[2:], nt, nn)
        return all(fits(y, nt, nn) for y in x)
    return x.__class__ is not int or x < max(nt, 1)


class Reworded:
    """a reader of templates whose kinds of sentences are said in other words (the places for things stay as they
    were, in the same order): to try a task whose meaning is known in a wording never met"""

    def __init__(self, reader, ways, way):
        self.reader, self.ways, self.way, self.guessed = reader, ways, way, reader.guessed

    def to(self, op, quest):
        slots = [c for c in op if c in "*#"]
        good = [w.lower().strip() for w in (self.ways.get("questions" if quest else "statements") or {}).get(op) or []
                if w.__class__ is str and [c for c in w if c in "*#"] == slots and re.fullmatch(r"[a-z*# ]+", w.lower().strip())]
        return good[self.way % len(good)] if good else op

    def rows(self, stories):
        return [[(self.to(op, quest), quest, things, numbers, kinds, gold) for op, quest, things, numbers, kinds, gold in ep] for ep in self.reader.rows(stories)]


def study(net, reader, raw, later, deep=True):
    """a task from a few stories: what the network names is checked exactly on them, kind of question by kind; what
    the check does not confirm is left to the search of recall.py (deep) -> the learner, how many kinds of questions
    were named rightly at once, of how many"""
    task = soft.widened(reader, recall.Task(reader.rows(raw), False), later)
    for ep in reader.rows(later):                                # (the kinds of questions to come are known by their words too)
        for op, quest, things, numbers, kinds, gold in ep:
            if quest:
                task.asked.setdefault(task.form(op, quest, things, kinds), len(task.asked))
    parts, wants = named(net, task) if net is not None else ({s: {} for s in SORTS}, {q: [] for q in task.asked.values()})
    m = recall.Learner.__new__(recall.Learner)
    m.task, m.readings, at_once, met = task, {}, 0, 0
    m.count = {q: len(task.golds(q)) for q in task.asked.values()}
    m.usual = {q: (Counter(task.golds(q)).most_common(1) or [("", 0)])[0][0] for q in task.asked.values()}
    m.own = {q: task.own(q) for q in task.asked.values()}
    for q in task.asked.values():
        best = None
        for sort, ask in wants[q]:
            got = recall.grade(recall.run(task, q, sort, ask, parts[sort], m.own[q]))
            if best is None or got[0] > best.right:
                best = recall.Reading(sort, ask, dict(parts[sort]), got[1], got[2], got[0], m.count[q])
            if got[0] == m.count[q]:
                break
        if m.count[q]:
            met += 1
            at_once += best is not None and best.right == best.total
            if (best is None or best.right < best.total) and deep:
                found = recall.learn(task, q)
                best = found if found and (best is None or found.right > best.right) else best
        m.readings[q] = best
    return m, at_once, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--found", default="", help="what the search of recall.py found (the files of --dump, comma-separated)")
    ap.add_argument("--hold", default="", help="sets the network is not taught on: K,L,M2")
    ap.add_argument("--reworded", default="", help="other wordings of the sentences of the lessons (JSON files, comma-separated)")
    ap.add_argument("--out", default="")
    ap.add_argument("--net", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--stories", type=int, default=20)
    ap.add_argument("--text", type=int, default=0, help="so many more stories are read with their answers hidden")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--alone", action="store_true", help="no search: what the check does not confirm stays as named")
    ap.add_argument("--plain", action="store_true", help="the stories as they are: no guess about their words")
    ap.add_argument("--reword", default="", help="other wordings of the sentences of the tasks tried (JSON files): the tasks are said in them")
    ap.add_argument("--way", type=int, default=0, help="which of the other wordings of a sentence")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.found:
        hold = set(a.hold.split(","))
        found = [json.loads(line) for f in a.found.split(",") for line in open(f)]
        mine = [x for x in found if os.path.basename(x["data"].rstrip("/")) not in hold]
        reworded = {x["id"]: x for f in a.reworded.split(",") if f for x in json.load(open(f))}
        lessons = [lesson(x, reworded.get(os.path.basename(x["data"].rstrip("/")) + str(x["task"]))) for x in mine]
        print(f"{len(mine)} tasks to teach on ({sum(len(s) for s, _, _ in lessons)} kinds of statements, {sum(len(q) for _, q, _ in lessons)} kinds of questions; "
              f"{sum(len(w) for _, _, other in lessons for w in other.values())} other wordings of them)", flush=True)
        net = teach(lessons, a.steps)
        right = sum(all(net.asks[i] == q for i, (_, _, q) in zip(net(*worded(one))[1].argmax(-1).tolist(), one[1])) for one in lessons if one[0] and one[1])
        print(f"of its own lessons, every kind of question named rightly: {right} of {len(lessons)}", flush=True)
        torch.save({"vocab": sorted(net.vocab), "parts": {s: sorted(set(net.parts[s]) - {"None"}) for s in SORTS}, "asks": net.asks, "weights": net.state_dict()}, a.out)
        return
    net = None                                                   # (with no network: the search alone, in the same harness)
    if a.net:
        saved = torch.load(a.net)
        net = Namer(saved["vocab"], saved["parts"], saved["asks"])
        net.load_state_dict(saved["weights"])
        net.eval()
    said_anew = {x["id"]: x for f in a.reword.split(",") if f for x in json.load(open(f))}
    files = glob.glob(os.path.join(a.data, "qa*_train.txt"))
    tasks = [int(x) for x in a.tasks.split(",") if x] or sorted(int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1)) for f in files)
    for k in tasks:
        t0 = time.time()
        path = glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0]
        raw = rows.stories(path, a.stories)
        hidden = [[(words, None if gold is None else []) for words, gold in story] for story in rows.stories(path, a.stories + a.text)[a.stories:]] if a.text else []
        later = rows.stories(glob.glob(os.path.join(a.data, f"qa{k}_*test.txt"))[0])
        best = None
        for reader in rows.guesses(raw[:20] + hidden):           # the guess about the words under which most is named rightly at once
            if a.reword:
                reader = Reworded(reader, said_anew.get(os.path.basename(a.data.rstrip("/")) + str(k), {}), a.way)
            m, at_once, met = study(net, reader, raw, later, deep=False)
            right = sum(r.right for r in m.readings.values() if r)
            mark = right - 0.5 * met - 3 * bool(reader.guessed)
            if best is None or mark > best[0]:
                best = (mark, reader)
            if at_once == met or a.plain:
                break
        reader = best[1]
        m, at_once, met = study(net, reader, raw, later, deep=not a.alone)
        right, total = sum(r.right for r in m.readings.values() if r), sum(m.count.values())
        full = sum(bool(r) and r.right == r.total and r.total >= 3 for r in m.readings.values())
        print(f"task {k:2d}: from {a.stories} stories, kinds of questions named rightly at once {at_once} of {met}" + (" - named at once" if at_once == met else "")
              + f"; {right}/{total} of their questions right ({time.time() - t0:.1f} s)" + (" - " + "; ".join(reader.guessed) if reader.guessed else ""), flush=True)
        r, t, sr, st = m.test(reader.rows(later))
        print(f"test {k:2d}: {r}/{t} = {100 * r / max(1, t):.1f}%; sure of {full} of {met} kinds of questions, {sr}/{st} right on them", flush=True)


if __name__ == "__main__":
    main()
