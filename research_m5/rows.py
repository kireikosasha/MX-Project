#!/usr/bin/env python3
"""
Template language as records: a sentence is a row. Its things are the words that take turns in one place of a
sentence; the words that stay are its operation. After that the learner of records (records.py) learns the task as
it learns a log of codes - the same state, the same moves and walks, nothing about words.

What a thing is, is found from the stories themselves: the words of the answers and the words that take turns in
a place of a question are things; a word that takes turns with a thing in a place of a statement is a thing too, and
of the same kind. For the other words that take turns somewhere it is tried what explains the stories better: that
they are things too (people who are never asked about, verbs that mean the same), or that they point at those the
sentence before began with (he, she, they).

    python rows.py --data tasks_1-20_v1-2/en --tasks 1,2,6,7,8
"""
import argparse
import glob
import os
import re
import time

import records

records.Learner.words = True                                     # sentences: different words may say the same

NUMBERS = {"none": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUMBERS.update({"monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4, "friday": 5, "saturday": 6, "sunday": 7,           # words that name a place
                "yesterday": 0, "morning": 1, "afternoon": 2, "evening": 3, "night": 4, "tomorrow": 5})                           # in an order are numbers too
PLAIN = ("yes", "no", "maybe", "nothing")
ODD = {"mice": "mouse", "men": "man", "women": "woman", "children": "child", "geese": "goose", "feet": "foot", "teeth": "tooth"}


def forms(data):
    """the words of the stories -> {a word in the plural: the same word in the singular}, for the words met in both forms"""
    seen = {w for story in data for words, gold in story for w in words + tuple(gold or ())}
    out = {}
    for w in seen:
        for one in (ODD.get(w), w[:-1] if w.endswith("s") else None, w[:-3] + "f" if w.endswith("ves") else None, w[:-2] if w.endswith("es") else None):
            if one and one in seen and one != w:
                out[w] = one
                break
    return out


def stories(path, limit=None):
    """a bAbI file -> [[(words, the answer or None)]]"""
    out = []
    for line in open(path, encoding="utf-8"):
        num, rest = line.strip().split(" ", 1)
        if num == "1":
            if limit and len(out) >= limit:
                break
            out.append([])
        parts = rest.split("\t")
        words = [str(NUMBERS[w]) if w in NUMBERS else w for w in re.findall(r"[a-z0-9]+", parts[0].lower())]
        gold = None
        if len(parts) > 1:
            gold = [str(NUMBERS[w]) if w in NUMBERS else w for w in parts[1].strip().lower().split(",")]
        out[-1].append((tuple(words), gold))
    return out


def names(data):
    """pairs of words that take turns with one word in a place of a question: two words that name one thing (the pink rectangle)"""
    slots = {}
    for story in data:
        for words, gold in story:
            for n in (1, 2) if gold is not None else ():
                for i in range(len(words) - n + 1):
                    if not any(w.isdigit() for w in words[i:i + n]):
                        slots.setdefault(words[:i] + ("*",) + words[i + n:], set()).add(words[i:i + n])
    return {x for spans in slots.values() if {len(x) for x in spans} == {1, 2} for x in spans if len(x) == 2}


def joined(data, pairs):
    """the stories with every such pair of words written as one word"""
    out = []
    for story in data:
        new = []
        for words, gold in story:
            ws, i = [], 0
            while i < len(words):
                if words[i:i + 2] in pairs:
                    ws.append("_".join(words[i:i + 2]))
                    i += 2
                else:
                    ws.append(words[i])
                    i += 1
            new.append((tuple(ws), gold))
        out.append(new)
    return out


class Reader:
    hint = None                                                  # the network that says which readings to try first, if there is one

    def __init__(self, train, limit=0):
        """which words are things, and of what kinds"""
        self.same = forms(train)
        self.raw, self.things, self.pointers = train, set(), set()
        self.ordered = False                                     # whether the order of the words of an answer says something
        train = self.plain(train)
        places = {}                                              # (question?, the sentence with one word out) -> the words met there
        things, said = set(), set()
        for story in train:
            for words, gold in story:
                if gold:
                    things.update(w for w in gold if w not in PLAIN and not w.isdigit())
                    said.update(gold)
                for i, w in enumerate(words):
                    if not w.isdigit():
                        places.setdefault((gold is not None, words[:i] + ("*",) + words[i + 1:]), set()).add(w)
        places = {k: v for k, v in places.items() if len(v) > 1}
        for (quest, _), ws in places.items():                    # what takes turns in a place of a question
            if quest:
                things |= ws
        self.kind = {}                                           # words that take turns in one place are of one kind
        for ws in places.values():
            ws = sorted(ws)
            for w in ws:
                self.kind.setdefault(w, w)
            for w in ws[1:]:
                self.kind[self.root(w)] = self.root(ws[0])
        for w in things:
            self.kind.setdefault(w, w)
        kinds = {}
        for w in self.kind:
            kinds.setdefault(self.root(w), set()).add(w)
        self.things = set()
        for ws in kinds.values():                                # what takes turns with a thing is a thing
            if ws & things:
                self.things |= ws
        self.name()
        self.guessed = []
        self.best = self.total = None
        if limit:                                                # the other kinds of words: by what explains more
            rest = [ws for ws in kinds.values() if not ws & self.things and len(ws) > 1]
            rest.sort(key=lambda ws: len({self.op(words, self.things | ws) for story in train for words, gold in story}))
            total = sum(gold is not None for story in train for words, gold in story)
            seeded = [ws for ws in kinds.values() if ws <= self.things and not ws & said]
            best = self.worth(limit)
            for ws in rest[:4]:
                if best[1] == total:                             # nothing is left to explain
                    break
                self.things |= ws                                # are they things too (people never asked about, verbs that mean the same)
                self.name()
                as_things = self.worth(limit)
                self.things -= ws
                self.name()
                self.pointers |= ws                              # or do they point at those the sentence before began with (he, she, they)
                as_pointers = self.worth(limit)
                self.pointers -= ws
                if max(as_things, as_pointers) > best:
                    best = max(as_things, as_pointers)
                    if as_things >= as_pointers:
                        self.things |= ws
                        self.name()
                    else:
                        self.pointers |= ws
                    self.guessed.append(("things: " if as_things >= as_pointers else "pointers: ") + " ".join(sorted(ws)))
            for ws in sorted(seeded, key=sorted):
                if best[1] == total:
                    break
                before = self.sentences()
                self.things -= ws                                # words that take turns in a question and are never an answer:
                self.name()                                      # things, or words of the question itself (who gave, who received)
                apart = self.worth(limit)
                if apart > best and 2 * (apart[1] - best[1]) >= self.sentences() - before:   # more kinds of sentences are more to learn:
                    best = apart                                 # a question more explained for every two new kinds, or it is not taken
                    self.guessed.append("words of the sentence itself: " + " ".join(sorted(ws)))
                else:
                    self.things |= ws
                    self.name()
            if best[1] < total and any(gold and len(gold) > 1 and list(gold) != sorted(gold) for story in train for words, gold in story):
                self.ordered = True                              # answers of several words: a set of things, or a sequence (the steps of a way)
                ordered = self.worth(limit)
                if ordered > best:
                    best = ordered
                    self.guessed.append("the order of the words of an answer matters")
                else:
                    self.ordered = False
            self.best, self.total = best, total

    def sentences(self):
        """how many kinds of sentences the stories have, as they are read now"""
        return len({(self.op(words, self.code), gold is None) for story in self.plain(self.raw) for words, gold in story})

    def root(self, w):
        while self.kind[w] != w:
            w = self.kind[w]
        return w

    def name(self):
        """a code for every thing, as in a log: the letter is the kind"""
        roots = sorted({self.root(w) for w in self.things})
        self.code = {}
        for w in sorted(self.things):
            letter = "abcdefghijklmnopqrstuvwxyz"[roots.index(self.root(w)) % 26]
            self.code[w] = letter + str(1 + sum(1 for c in self.code.values() if c[0] == letter))

    @staticmethod
    def op(words, things):
        return " ".join("#" if w.isdigit() else "*" if w in things else w for w in words)

    def worth(self, limit):
        m = records.Learner(self.rows(self.raw))
        m.hint = self.hint
        m.learn(limit)
        return m.fit(m.rec)[0]

    def plain(self, data):
        """every word in its singular form; a pointer gives way to those the sentence before began with"""
        f, out = self.same, []
        for story in data:
            new, last = [], ()
            for words, gold in story:
                words = tuple(f.get(w, w) for w in words)
                if last and self.pointers.intersection(words):
                    words = tuple(x for w in words for x in (last if w in self.pointers else (w,)))
                if gold is None:
                    last = self.lead(words)
                new.append((words, gold and [f.get(w, w) for w in gold]))
            out.append(new)
        return out

    def lead(self, words):
        """the things a sentence begins with (mary; mary and daniel), or its first thing"""
        if words and words[0] in self.things:
            i = 1
            while i + 1 < len(words) and words[i] not in self.things and words[i + 1] in self.things:
                i += 2
            return words[:i]
        return tuple(w for w in words if w in self.things)[:1]

    def rows(self, data):
        """stories -> episodes as records.py reads them"""
        out = []
        for story in self.plain(data):
            ep = []
            for words, gold in story:
                things = tuple(self.code[w] for w in words if w in self.code)
                numbers = tuple(int(w) for w in words if w.isdigit())
                kinds = tuple("#" if w.isdigit() else records.kind(self.code[w]) for w in words if w.isdigit() or w in self.code)
                if gold is not None:
                    gold = [self.code.get(w, w) for w in gold]
                    gold = ",".join(gold if self.ordered else sorted(gold))
                ep.append((self.op(words, self.code), gold is not None, things, numbers, kinds, gold))
            out.append(ep)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a folder with bAbI files (qaN_*_train.txt, qaN_*_test.txt)")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--stories", type=int, default=20)
    ap.add_argument("--limit", type=float, default=records.WORK)
    ap.add_argument("--ask", type=int, default=0, help="how many more stories it may ask the answers of (those on which its readings part)")
    ap.add_argument("--more", default="", help="take more stories where what is learnt fails on fresh ones: e.g. 60,200")
    ap.add_argument("--hint", default="", help="a network that says which readings of a row to try first (made by prior.py)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    hint = None
    if a.hint:
        import prior
        hint = Reader.hint = prior.Hint(a.hint)
    files = glob.glob(os.path.join(a.data, "qa*_train.txt"))
    tasks = [int(x) for x in a.tasks.split(",") if x] or sorted(int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1)) for f in files)
    right = total = 0
    for k in tasks:
        t0 = time.time()
        more = [int(x) for x in a.more.split(",") if x]
        raw = stories(glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0], max([a.stories + 200 * bool(a.ask)] + more))
        reader, pairs = None, names(raw[:a.stories])
        if pairs:                                                # some things may be named by two words (the pink rectangle): tried only
            plain = Reader(raw[:a.stories], 0)                   # if a real part of the stories is not explained as they are, and first
            if 10 * plain.worth(a.limit / 4)[1] < 9 * sum(gold is not None for story in raw[:a.stories] for words, gold in story):   # (a tenth and more)
                other = Reader(joined(raw[:a.stories], pairs), a.limit / 4)
                if other.best[1] == other.total:                 # taken only if everything is explained so
                    reader, raw = other, joined(raw, pairs)
                    reader.guessed.insert(0, "two words name one thing: " + " ".join(sorted("_".join(x) for x in pairs)))
                else:
                    pairs = ()
            else:
                pairs = ()
        if reader is None:
            reader = Reader(raw[:a.stories], a.limit / 4)         # what a thing is, is settled on the first stories
        m, used, notes = records.study(reader.rows(raw), a.stories, more, a.limit, a.ask, hint=hint)
        got, n = m.run(m.episodes)
        print(f"task {k:2d}: {len(reader.code)} things of {len({c[0] for c in reader.code.values()})} kinds; learnt from {used} stories" + "".join(f" ({x})" for x in notes) + f", {got}/{n} of their "
              f"questions right, {sum(bool(p) for p in m.rec.values())} of {len(m.rec)} kinds of statements and {len(m.ask)} of "
              f"{len(m.forms) - len(m.rec)} kinds of questions read ({time.time() - t0:.0f} s, {m.spent} episodes played)", flush=True)
        if reader.guessed:
            print("  taken by what they explain -", "; ".join(reader.guessed), flush=True)
        if a.show:
            print("  things:", " ".join(f"{w}={c}" for w, c in sorted(reader.code.items(), key=lambda x: x[1])))
            print(m.describe(), flush=True)
        r, t = m.run(reader.rows(joined(stories(glob.glob(os.path.join(a.data, f"qa{k}_*test.txt"))[0]), pairs)))
        right, total = right + r, total + t
        print(f"test {k:2d}: {r}/{t} = {100 * r / t:.1f}%", flush=True)
    if total:
        print(f"all: {right}/{total} = {100 * right / total:.2f}%", flush=True)


if __name__ == "__main__":
    main()
