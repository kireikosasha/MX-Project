#!/usr/bin/env python3
"""
Learning by trying and checking: from stories with answers, and nothing else, find out what every sentence
does and what every question asks.

Given: the ground (core.py), which knows no word of any language and no subject. Not given: a vocabulary, a
grammar, a list of verbs, which words name things, which question is about what.

How it learns:
1. Places that vary. If two sentences differ in one place only, that place holds something replaceable (a
   name, a place, a verb...). Such words are the candidates for being arguments.
2. A guess for a sentence: one of the schemes of the ground, a number of an attribute, and which candidate
   places are its arguments. The other words of the sentence become the fixed part of the form, so a guess for
   one sentence is a guess for every sentence of the same form.
3. Checking. The stories are run on the ground with the guess; the guess that makes more answers right is
   kept. A question and a statement are guessed together when neither is known yet.
4. Single words are learnt the same way: which words are pronouns, which word is which direction, the order of
   the times of day, which two forms are one word, how the symbols of the ground (yes, no, numbers, steps of a
   path) are said in this language, and what usually follows from what.

    python acquire.py --data tasks_1-20_v1-2/en --stories 20
Tasks are learnt one after another, and what was learnt stays.
"""
import argparse
import glob
import itertools
import json
import os
import pickle
import sys
import re
import time
from collections import Counter, defaultdict

import core as G

STAR = "\u2605"                  # an argument place of a form
STARS = STAR + STAR               # an argument made of two or more places next to each other ("red square")


def read(path, limit=None):
    """a bAbI file -> stories: [(words, answer or None)]; an answer is a list (several things, or the steps of a path)"""
    out = []
    for line in open(path, encoding="utf-8"):
        n, rest = line.strip().split(" ", 1)
        if n == "1":
            if limit and len(out) >= limit:
                break
            out.append([])
        parts = rest.split("\t")
        words = tuple(re.findall(r"[^\W\d_]+|[.,?]", parts[0].lower()))
        out[-1].append((words, [x.strip() for x in parts[1].lower().split(",")] if len(parts) > 1 else None))
    return out


def alike(a, b):
    """do two words look like forms of one word (a guess, to be checked by answers)"""
    if a == b or " " in a or " " in b:
        return 0
    pre = len(os.path.commonprefix([a, b]))
    if pre >= 3 and abs(len(a) - len(b)) <= 3:
        return 10 + pre
    if len(a) >= 4 and a[1:] == b[1:]:                         # only the first letter differs (as a capital letter may)
        return 9
    if a[0] == b[0] and a[-1] == b[-1] and abs(len(a) - len(b)) <= 1 and len(a) >= 4:
        return 5
    return 0


class Mind:
    def __init__(self):
        self.lex = G.Lexicon()
        self.forms = {}                    # form -> (scheme, number, the role of every argument place)
        self.verbs = {}                    # a word in a place that is not an argument -> (scheme, number): what the word itself means
        self.bad = set()                   # guesses that were tried and spoiled something
        self.frozen = set()                # the forms and words of the guess being defended while earlier tasks are re-read
        self.single, self.multi = set(), set()
        self.said = {}                     # a symbol of the ground -> how it is said
        self.said_n = defaultdict(Counter)
        self.said_all = {}                 # every saying learnt, also from tasks that were not learnt well
        self.maybe_alias = {}              # forms that look like one word: tried when a guess needs them
        self.sort = {}                     # a word -> the sort of words it can be replaced with
        self.apart = set()                 # pairs of words seen in one place
        self.joint = set()                 # several words seen standing together where one word can stand
        self.seen = []                     # the stories learnt from
        self.tasks = []                    # the same stories, task by task
        self._keys, self._parse, self._by_form = {}, {}, defaultdict(set)
        self._text, self._named, self._read_by = {}, {}, {}
        self.trace = False

    # ---------------------------------------------------------------- places that vary
    def sort_of(self, filler):
        """the sort of a word or of several words standing as one (a tuple)"""
        filler = filler[0] if len(filler) == 1 else tuple(filler)
        return self.sort.get(filler, filler)

    def same(self, word):
        """a word, or the word it may be a form of ("mice" - "mouse"): to see that an answer is named in a story"""
        return self.lex.alias.get(word) or self.maybe_alias.get(word, word)

    def find_places(self, stories):
        golds = {g for story in stories for _, gold in story for g in (gold or ())}
        sents = sorted({(words, gold is not None) for story in stories for words, gold in story})   # in a fixed order
        apart = self.apart                                     # pairs seen in one place: they are of one sort, and are different things

        def sorts():
            self.sort = {x: x for pair in sorted(apart, key=repr) for x in pair}

            def top(w):
                while self.sort[w] != w:
                    w = self.sort[w]
                return w
            for a, b in sorted(apart, key=repr):
                self.sort[top(a)] = top(b)
            self.sort = {w: top(w) for w in self.sort}

        for rnd in range(4):
            # first with the words as they are; then again with the places already found replaced by their sort,
            # so that a rare word shows that it varies too
            keys = defaultdict(set)
            for words, quest in sents:
                seen, pos = [], 0                              # (the words, is it a place already known)
                for s0, e0 in (self.places(words) if rnd else ()):
                    seen += [((w,), False) for w in words[pos:s0]] + [(tuple(words[s0:e0]), True)]
                    pos = e0
                seen += [((w,), False) for w in words[pos:]]
                around = [("sort", self.sort_of(w)) if k else w for w, k in seen]
                n = len(seen)
                while n and seen[n - 1][0][0] in ".?,":
                    n -= 1
                for i in range(n):
                    for j in range(i + 1, min(n, i + 4) + 1):
                        if j - i > 1 and any(k for _, k in seen[i:j]):
                            continue                           # a known place is looked at alone
                        filler = tuple(x for w, _ in seen[i:j] for x in w)
                        if len(filler) <= 4:
                            keys[(quest, tuple(around[:i]), tuple(around[j:]))].add(filler)
            size = (len(self.single), len(self.multi), len(apart))
            for fillers in keys.values():
                if len(fillers) < 2 or len({f[0] for f in fillers}) < 2 or len({f[-1] for f in fillers}) < 2:
                    continue                               # the same word at an edge: the real place is smaller
                ones = sorted(f[0] for f in fillers if len(f) == 1)
                if not ones:
                    continue
                self.single |= set(ones)
                self.joint |= {f for f in fillers if len(f) > 1}
                many = [f for f in fillers if len(f) > 1 and not any(w in self.single for w in f[1:])]
                self.multi |= set(many)                    # several words standing where one word can stand
                apart |= {(a, b) for a in ones + many for b in ones + many if a != b}
            known = self.single | self.multi
            self.multi = {f for f in self.multi if not ((f[1:] if len(f) > 2 else f[1]) in known or (f[:-1] if len(f) > 2 else f[0]) in known)
                          and not any(w in self.single for w in f[1:])}      # a word that stands alone, inside: not one thing
            sorts()
            if (len(self.single), len(self.multi), len(apart)) == size:
                break
        # words that look alike and never stand against each other in one place may be one word
        words = sorted(self.single | {g for g in golds if " " not in g})
        group = {w: {w} for w in words}
        pairs = sorted(((alike(a, b), a, b) for i, a in enumerate(words) for b in words[i + 1:] if alike(a, b)), reverse=True)
        for _, a, b in pairs:
            ga, gb = group[a], group[b]
            if ga is not gb and not any((x, y) in apart for x in ga for y in gb):
                ga |= gb
                for x in gb:
                    group[x] = ga
        self.maybe_alias = {}
        for w, g in group.items():
            if len(g) > 1:
                head = min(g, key=lambda x: (x not in golds, len(x), x))
                if w != head:
                    self.maybe_alias[w] = head
        self._keys.clear()
        self._parse.clear()
        self._by_form.clear()

    def places(self, words):
        """the candidate places of a sentence: [(start, end)]"""
        out, i = [], 0
        while i < len(words):
            for size in (4, 3, 2):
                if words[i:i + size] in self.multi:
                    out.append((i, i + size))
                    i += size
                    break
            else:
                if words[i] in self.single:
                    out.append((i, i + 1))
                i += 1
        return out

    # ---------------------------------------------------------------- forms
    def keys(self, words, quest):
        """every way to see a sentence as a form: [(form, the argument stretches)]"""
        if (words, quest) in self._keys:
            return self._keys[(words, quest)]
        pl = self.places(words)[:7]
        out = []
        for mark in itertools.product((0, 1, 2), repeat=len(pl)):       # 0 fixed, 1 an argument, 2 an argument joined to the previous one
            if sum(m == 1 for m in mark) > 4:                          # at most four arguments
                continue
            ok = all(m != 2 or (i > 0 and mark[i - 1] > 0 and pl[i - 1][1] == pl[i][0]) for i, m in enumerate(mark))
            if not ok:
                continue
            runs = []
            for (s, e), m in zip(pl, mark):
                if m == 1:
                    runs.append([s, e, STAR])
                elif m == 2:
                    runs[-1][1:] = [e, STARS]
            form, pos = [quest], 0
            for s, e, star in runs:
                form += list(words[pos:s]) + [star]
                pos = e
            form += list(words[pos:])
            out.append((tuple(form), tuple((s, e) for s, e, _ in runs)))
        self._keys[(words, quest)] = out
        for form, _ in out:
            self._by_form[form].add((words, quest))
        return out

    def parse(self, words, quest):
        """-> (scheme, number, roles) or None"""
        hit = self._parse.get((words, quest), 0)
        if hit != 0:
            return hit
        best = None
        for form, runs in self.keys(words, quest):                      # the form with more fixed words wins
            f = self.forms.get(form)
            if f and (best is None or (len(runs), f[3]) < (len(best[1]), best[0][3])):
                best = (f, runs)
        out = None
        self._read_by[(words, quest)] = None
        for form, runs in (self.keys(words, quest) if best else ()):
            if self.forms.get(form) is best[0] and runs == best[1]:
                self._read_by[(words, quest)] = form
        if best:
            (kind, k, roles, _), runs = best
            a = {}
            for r, (s, e) in zip(roles, runs):
                if r != "O":
                    a.setdefault(r, []).append(" ".join(words[s:e]))
                elif not quest and " ".join(words[s:e]) in self.verbs:       # the word in this place says what is done
                    v = self.verbs[" ".join(words[s:e])]
                    need, may, _ = G.KINDS[v[0]]
                    if set(need) <= set(roles) - {"O"} <= set(need) | set(may):
                        kind, k = v
            out = (kind, k, a)
        self._parse[(words, quest)] = out
        return out

    def guesses(self, words, quest, again=False, gold=None, text=()):
        """every guess for a sentence: [(form, (scheme, number, roles))]. again: also other guesses for forms already
        known. gold: the answer, when the sentence is a question - the kind of answer tells the kind of question"""
        used = {"a": {f[1] for f in self.forms.values() if G.KINDS[f[0]][2] == "a"},
                "o": {f[1] for f in self.forms.values() if G.KINDS[f[0]][2] == "o"},
                "m": {f[1] for f in self.forms.values() if G.KINDS[f[0]][2] == "m"}}
        room = {"a": G.N_ATTR, "o": G.N_ORDER, "m": G.N_MAP}
        kinds = G.QUERIES if quest else G.STATEMENTS
        if quest and gold:
            things = ("VAL", "BEFORE", "P_WHAT", "P_FROM", "P_TO", "NEXT", "L_A1", "L_A2", "LIST")
            signs = ("IS", "MORE", "L_IS", "COUNT", "LIST", "PATH")
            named = {self.same(w) for w in gold[0].split()} <= {self.same(w) for w in text}
            if len(gold) > 1:
                can = ("LIST", "PATH")
            elif named and any(gold[0] in c for c in self.said_n.values()):
                can = signs + things          # named in the story, but already learnt as a way to say yes, no or a number
            elif named:                       # the answer is named in the story: a thing is asked
                can = things
            else:                             # it is not: yes or no, a number, a step
                can = signs
            kinds = {k: v for k, v in kinds.items() if k in can}
        first, out = [], []
        for form, runs in self.keys(words, quest):
            if (form in self.forms and not again) or not runs:
                continue
            # by likeness: a form already known with one more place (a time, a word that does not matter)
            stars = [i for i, w in enumerate(form) if w in (STAR, STARS)]
            if STARS in form:                                       # ... or with two places where it has one ("red square" for "triangle")
                for like in sorted({form[:at] + (STAR,) + form[at + 1:] for at in stars} | {tuple(STAR if w == STARS else w for w in form)}, key=repr):
                    known = self.forms.get(like)
                    if like != form and known and known[0] in kinds:
                        first.append((form, known[:3]))
            for j, at in enumerate(stars):
                for before, after in ((0, 0), (1, 0), (2, 0), (0, 1)):
                    cut = range(at - before, at + after + 1)
                    if at - before < 1 or at + after >= len(form) or any(form[i] in (STAR, STARS) for i in cut if i != at):
                        continue
                    known = self.forms.get(tuple(w for i, w in enumerate(form) if i not in cut))
                    if known and known[0] in kinds:
                        kind, k, roles, _ = known
                        for r in [m for m in G.KINDS[kind][1] if m not in roles] + (["O"] if "O" not in roles else []):
                            first.append((form, (kind, k, roles[:j] + (r,) + roles[j:])))
            for kind, (need, may, num) in kinds.items():
                for extra in ([()] + [(m,) for m in may]):
                    want = need + extra
                    if len(runs) < len(want) or len(runs) > len(want) + 2:
                        continue
                    for roles in sorted(set(itertools.product(want + ("O",), repeat=len(runs)))):
                        if set(roles) - {"O"} != set(want) or any(roles.count(r) > 1 for r in ("D", "T", "O")) \
                                or sum(roles.count(r) > 1 for r in ("A1", "A2", "A3")) > 1:
                            continue
                        if num == "":
                            ks = [0]
                        elif num == "aa":
                            top = min(G.N_ATTR, len(used["a"]) + 1)
                            ks = [(i, j) for i in range(top) for j in range(top) if i != j]
                        else:                                    # a new number first: what is new is not mixed with what is known
                            new = len(used[num])
                            ks = ([new] if new < room[num] else []) + list(range(min(new, room[num])))
                        out += [(form, (kind, k, roles)) for k in ks]
        out.sort(key=lambda g: (len(g[1][2]), len(g[1][2]) - len(set(g[1][2]))))      # the simpler guess first
        self._like = set(first)
        self._like_join = {g for g in first if STARS in g[0] and self.forms.get(tuple(STAR if w == STARS else w for w in g[0]), ())[:3] == g[1]}
        seen, both = set(), []
        for g in first + out:
            if g not in seen and not (again and self.forms.get(g[0], ())[:3] == g[1]):
                seen.add(g)
                both.append(g)
        return both

    # ---------------------------------------------------------------- checking
    def run(self, story, upto=None):
        """-> the answers of the ground to the questions of a story: [(answer, the right one)]"""
        w, out = G.World(self.lex), []
        if id(story) not in self._text:
            self._text[id(story)] = {x for ws, _ in story for x in ws}
        for words, gold in story[:upto]:
            if gold is not None and id(gold) not in self._named:      # is the answer something named in the story
                self._named[id(gold)] = set(" ".join(gold).split()) <= self._text[id(story)]
            p = self.parse(words, gold is not None)
            if gold is None:
                if p:
                    try:
                        w.tell(*p)
                    except (IndexError, KeyError):
                        pass
                continue
            ans = None
            if p:
                try:
                    ans = w.ask(*p)
                except (IndexError, KeyError):
                    ans = None
            out.append((ans, gold))
        return out

    def judge(self, results, learn=False, fixed=False):
        """how many answers are right, with the symbols of the ground said in the way that fits best. A symbol (yes,
        no, a number, nothing, a step) is never said with the name of a thing of the story. fixed: the symbols whose
        saying is already learnt keep it"""
        count = defaultdict(Counter)
        for n in (self.said_n, ):
            for s, c in n.items():
                count[s].update(c)
        now = defaultdict(Counter)
        for ans, gold in results:
            if self._named.get(id(gold)):
                continue
            if ans and ans[0] == "s" and len(gold) == 1:
                now[ans[1]][gold[0]] += 1
            elif ans and ans[0] == "p" and len(ans[1]) == len(gold):
                for step, g in zip(ans[1], gold):
                    now["step " + step][g] += 1
        for s, c in now.items():
            count[s].update(c)
        said = dict(self.said) if fixed else {}
        taken = set(said.values())
        for s in sorted((s for s in count if s not in said), key=lambda s: -max(count[s].values())):
            for g, _ in count[s].most_common():
                if g not in taken:
                    said[s] = g
                    taken.add(g)
                    break
        if learn:
            self.said = said
        return sum(self.right(ans, gold, said) for ans, gold in results), said

    def right(self, ans, gold, said):
        if ans is None:
            return False
        t, v = ans
        if t == "e":
            return len(gold) == 1 and self.lex.alias.get(gold[0], gold[0]) == v
        if t == "l":
            return sorted(v) == sorted(self.lex.alias.get(g, g) for g in gold)
        if t == "s":
            return len(gold) == 1 and said.get(v) == gold[0]
        return len(v) == len(gold) and all(said.get("step " + s) == g for s, g in zip(v, gold))

    def flat(self, forms, lex, stories):
        """does some question form of a guess give always the same answer while the right answers differ"""
        mine = {f for f, _ in forms if f[0]}
        if not mine:
            return False
        got = defaultdict(lambda: (set(), set()))
        with self.with_forms(forms, lex):
            for story in stories:
                res, qi = self.run(story), -1
                for words, gold in story:
                    if gold is None:
                        continue
                    qi += 1
                    f = self._read_by.get((words, True))
                    if f in mine and res[qi][0] is not None:
                        got[f][0].add(str(res[qi][0]))
                        got[f][1].add(str(gold))
        return any(len(a) < 2 <= len(g) for a, g in got.values())

    def score(self, stories):
        res = [r for s in stories for r in self.run(s)]
        return self.judge(res)[0], len(res)

    def with_forms(self, new, lex=None, verbs=None):
        """a context: the forms with some added or replaced, and maybe other single words"""
        mind = self

        class Trial:
            def __enter__(self):
                self.old = {f: mind.forms.get(f) for f, _ in new}
                self.lex = mind.lex
                self.verbs = mind.verbs
                if verbs:
                    mind.verbs = dict(mind.verbs, **verbs)
                    mind._parse.clear()
                for f, h in new:
                    mind.forms[f] = h + (len(mind.forms) if f not in mind.forms else mind.forms[f][3],)
                if lex is not None:
                    mind.lex = lex
                self.forget()

            def forget(self):
                for f, _ in new:
                    for sent in mind._by_form.get(f, ()):
                        mind._parse.pop(sent, None)

            def __exit__(self, *a):
                for f, h in self.old.items():
                    if h is None:
                        del mind.forms[f]
                    else:
                        mind.forms[f] = h
                mind.lex = self.lex
                if verbs:
                    mind.verbs = self.verbs
                    mind._parse.clear()
                self.forget()
        return Trial()

    # ---------------------------------------------------------------- learning
    def snapshot(self):
        return dict(self.forms), self.lex.copy(), dict(self.said), dict(self.verbs)

    def restore(self, snap):
        self.forms, self.lex, self.said, self.verbs = dict(snap[0]), snap[1], dict(snap[2]), dict(snap[3])
        self._parse.clear()

    def take(self, new, lex, verbs=None):
        for f, h in new:
            self.forms[f] = h + (self.forms[f][3] if f in self.forms else len(self.forms),)
        if lex is not None:
            self.lex = lex
        self.verbs.update(verbs or {})
        self._parse.clear()

    def learn(self, stories, budget=600):
        """one more task: its stories with answers"""
        t0 = time.time()
        before = [self.score(t)[0] for t in self.tasks]            # what the earlier tasks give now: it must not get worse
        known_before = set(self.forms)
        self.seen += stories
        self.tasks.append(stories)
        self.find_places(self.seen)
        self.adopt(stories)
        best = self.score(stories)[0]
        total = sum(1 for s in stories for _, g in s if g is not None)
        stuck, whole = set(), True
        while best < total and time.time() - t0 < budget:
            step = self.try_task(stories, best, min(t0 + budget, time.time() + 0.5 * budget)) if whole else None   # first the task as a whole
            if step is None:
                whole = False
                step = self.next_step(stories, best, stuck, t0 + budget)
            if step is None:
                if not self.words(stories, best):
                    break
                best = self.score(stories)[0]
                stuck.clear()
                continue
            new, sc, lex, verbs, unit = step
            snap = self.snapshot()
            self.take(new, lex, verbs)
            if any(self.score(t)[0] < b for t, b in zip(self.tasks, before)):
                # the new understanding spoils an earlier task: read that task again in its light, or give the guess up
                self.frozen = {f for f, _ in new} | set(verbs or {})
                t1 = time.time()
                mended = self.repair(before, min(t0 + budget, t1 + 0.2 * budget))
                self.frozen = set()
                if self.trace:
                    print(f"   ? an earlier task got worse with: " + "; ".join(" ".join(map(str, f[1:])) + " = " + str(h) for f, h in new)
                          + (f" {verbs}" if verbs else "") + f" - {'mended' if mended else 'not mended'} in {time.time() - t1:.0f} s", flush=True)
                if not mended or self.score(stories)[0] <= best:
                    self.restore(snap)
                    self.bad.add((tuple(new), tuple(sorted((verbs or {}).items()))))
                    continue
            self.adopt(stories)
            best = self.score(stories)[0]
            if self.trace:
                print(f"   + {best}/{total} ({time.time() - t0:.0f} s): " + "; ".join(" ".join(map(str, f[1:])) + " = " + str(h) for f, h in new)
                      + (f" {verbs}" if verbs else ""), flush=True)
            self.judge([r for s in stories for r in self.run(s)], learn=True)
        self.words(stories, best)
        self.prune(set(self.forms) - known_before)
        best = self.score(stories)[0]
        res = [r for s in stories for r in self.run(s)]
        _, said = self.judge(res, learn=True)
        self.said_all.update({s: g for s, g in said.items() if s not in self.said_n})
        for ans, gold in (res if best >= 0.9 * total else ()):      # how the symbols are said stays for the next tasks
            if ans and ans[0] == "s" and len(gold) == 1 and said.get(ans[1]) == gold[0]:
                self.said_n[ans[1]][gold[0]] += 1
            elif ans and ans[0] == "p" and self.right(ans, gold, said):
                for s, g in zip(ans[1], gold):
                    self.said_n["step " + s][g] += 1
        return best, total

    def prune(self, new_forms):
        """what was learnt is kept only if the answers are worse without it: every pronoun, every meaning of a word and
        every form of this task is taken away in turn"""
        base = [self.score(t)[0] for t in self.tasks]

        def worse():
            self._parse.clear()
            return any(self.score(t)[0] < b for t, b in zip(self.tasks, base))
        for w in sorted(self.lex.pron):
            self.lex.pron.discard(w)
            if worse():
                self.lex.pron.add(w)
        for w in sorted(self.verbs):
            v = self.verbs.pop(w)
            if worse():
                self.verbs[w] = v
        for f in sorted(new_forms, key=lambda f: -self.forms[f][3] if f in self.forms else 0):
            if f in self.forms:
                h = self.forms.pop(f)
                if worse():
                    self.forms[f] = h
        self._parse.clear()

    def adopt(self, stories):
        for _ in range(3):                                       # what is taken may make something else alike
            n = len(self.forms)
            self.adopt_once(stories)
            if len(self.forms) == n:
                break

    def adopt_once(self, stories):
        """a statement that is not understood, but is a known form with one more varying word, is taken to mean the
        same (the word being a time, or nothing): a guess by likeness, kept unless it spoils something. Of several
        likenesses the one is taken that leaves the words in the roles they are known in"""
        was = [self.score(t)[0] for t in self.tasks]
        known = defaultdict(set)                                 # role -> the sorts of words seen in it

        def note(p):
            for r, vals in p[2].items():
                known[r] |= {self.sort_of(v.split()) for v in vals}
        for story in self.seen:
            for words, gold in story:
                p = self.parse(words, gold is not None)
                if p:
                    note(p)
        for story in stories:
            for words, gold in story:
                if self.parse(words, gold is not None) is not None:
                    continue
                if gold is not None:                            # a question: only the same question with an argument of two words
                    for g in self.guesses(words, True):
                        if g in self._like_join:
                            self.take([g], None)
                            break
                    continue
                things = known["A1"] | known["A2"] | known["A3"]
                like = []
                for n, g in enumerate(self.guesses(words, False)):
                    if g not in self._like:
                        break
                    runs = next(r for f, r in self.keys(words, False) if f == g[0])
                    fit = 0
                    for r, (s0, e0) in zip(g[1][2], runs):
                        w = self.sort_of(words[s0:e0])
                        fit += (w in known[r]) if r.startswith("A") else -(w in things)
                    like.append((-fit, n, g))
                for _, _, g in sorted(like):
                    with self.with_forms([g]):
                        fine = all(self.score(t)[0] >= b for t, b in zip(self.tasks, was))
                    if fine:
                        self.take([g], None)
                        note(self.parse(words, False))
                        break

    def repair(self, before, until):
        """earlier tasks answered worse than they were: look again at their sentences until they are as good as before"""
        for _ in range(2):
            for task, was in zip(self.tasks[:-1], before):
                cur, stuck = self.score(task)[0], set()
                while cur < was and time.time() < until:
                    step = self.next_step(task, cur, stuck, until, revise=True)
                    if step is None:
                        return False
                    self.take(step[0], step[2], step[3])
                    cur = self.score(task)[0]
            if all(self.score(t)[0] >= b for t, b in zip(self.tasks[:-1], before)):
                return True
        return False

    def next_step(self, stories, best, stuck, until, revise=False):
        """the first question that is answered wrong: guess what is not understood in its story, keep the best guess.
        revise: nothing new is read, only what is already understood is looked at again"""
        said = self.judge([r for s in stories for r in self.run(s)])[1]
        for si, story in enumerate(stories):
            res, qi = self.run(story), -1
            for li, (words, gold) in enumerate(story):
                if gold is None:
                    continue
                qi += 1
                if self.right(res[qi][0], gold, said) or (id(story), li) in stuck:
                    continue
                dark, read = [], []                              # what is not understood before this question, and all that was read
                for w2, g2 in story[:li + 1]:
                    one = (w2, g2 is not None)
                    if self.parse(*one) is None and one not in dark:
                        dark.append(one)
                for w2, g2 in reversed(story[:li]):
                    if (w2, g2 is not None) not in read and (w2, g2 is not None) not in dark and g2 is None:
                        read.append((w2, False))
                quest = (words, True)
                new = lambda d: d + (False,)                     # noqa: E731
                old = lambda d: d + (True,)                      # noqa: E731
                sets = []
                if not revise and quest in dark:
                    others = list(reversed([d for d in dark if d != quest][-5:]))
                    sets += [[new(quest)]] + [[new(quest), new(d)] for d in others]
                    sets += [[new(quest), old(d)] for d in read[:4]]
                    sets += [[new(quest), new(x), new(y)] for x, y in itertools.combinations(others, 2)][:6]
                elif not revise and dark:
                    others = list(reversed(dark[-6:]))
                    sets += [[new(d)] for d in others] + [[new(x), new(y)] for x, y in itertools.combinations(others, 2)][:6]
                if quest not in dark:                            # or something was understood wrongly: look at it again
                    sets += [[old(quest)]] + [[old(d)] for d in read[:6]] + [[old(quest), old(d)] for d in read[:4]]
                elif revise:
                    sets += [[new(quest)]] + [[new(quest), old(d)] for d in read[:4]]
                word = self.try_words(story, li, stories, best) if quest not in dark else None
                for group in sets:
                    if time.time() > until:
                        return None
                    if word and len(group) > 1:
                        continue                             # a meaning of a word is at hand: only a single better reading beats it
                    found = self.try_group(group, story, li, stories, word[1] if word else best, until, revise)
                    if found:
                        return found + (None, (id(story), li))
                if word:
                    return word + ((id(story), li),)
                stuck.add((id(story), li))
        return None

    def try_task(self, stories, best, until):
        """the task as a whole: guess what the kinds of statements that are not understood mean, and for every such
        guess choose the reading of each question on its own - given the statements, the questions do not depend on
        each other. Tried first with sentences of the same build read alike ("is above" like "is to the left of"),
        then freely"""
        def shape(words, quest, joins=True):
            """the form in which every varying place is an argument; places next to each other that were seen standing
            together where one word can stand ("red square" for "triangle") are one argument"""
            runs = []
            for s0, e0 in self.places(words):
                if runs and runs[-1][1] == s0 and tuple(words[runs[-1][0]:e0]) in self.joint:
                    runs[-1][1:] = [e0, STARS]
                else:
                    runs.append([s0, e0, STAR])
            form, pos = [quest], 0
            for s0, e0, star in runs:
                form += list(words[pos:s0]) + [star if joins else STAR]
                pos = e0
            return tuple(form + list(words[pos:]))

        dark_s, dark_q = {}, {}          # the build of a sentence -> one sentence of it, and the forms it is met in
        for story in stories:
            text = {w for ws, _ in story for w in ws}
            for words, gold in story:
                quest = gold is not None
                if self.parse(words, quest) is None:
                    kind = (dark_q if quest else dark_s).setdefault(shape(words, quest, False), {"rep": (words, gold, text), "forms": []})
                    if shape(words, quest) not in kind["forms"]:
                        kind["forms"].append(shape(words, quest))
        if not (dark_s or dark_q) or len(dark_q) > 12:
            return None

        def general(kind, quest, only=False):
            """the guesses for one build of sentence in which every varying place is an argument (at the level of the
            whole task no single word is yet taken as part of a form); a guess for the form of the sentence itself
            holds for every form its build is met in. only: no other guesses"""
            words, gold, text = kind["rep"]
            base, pl, runs_of = shape(words, quest), self.places(words), dict(self.keys(words, quest))
            out = []
            for f, h in self.guesses(words, quest, False, gold, text):
                if f == base:
                    out.append((0, [(x, h) for x in kind["forms"]]))
                elif not only and all(any(s0 <= a and b <= e0 for s0, e0 in runs_of[f]) for a, b in pl):
                    out.append((1 + (STARS in f), [(f, h)]))
            return [forms for _, forms in sorted(out, key=lambda x: x[0])]

        every, some = stories, stories[:10]                      # the search looks at ten stories, the result is checked on all
        n_all = sum(1 for s in every for _, g in s if g is not None)
        told = {p[0] for story in stories for w, g in story if g is None for p in [self.parse(w, False)] if p}
        fresh = {w: h for w, h in self.maybe_alias.items() if w not in self.lex.alias}
        lexes = [None]
        if fresh:
            lexes.append(self.lex.copy())
            lexes[-1].alias.update(fresh)
        loose_kinds = ("BEFORE", "PATH", "L_IS", "L_A1", "L_A2", "LINK")

        def total(forms, lex, where=some):
            with self.with_forms(forms, lex):
                return self.judge([r for s in where for r in self.run(s)])[0]

        def search(s_cands, q_cands, enough):
            """s_cands: the guesses for the statements, each a list of forms; q_cands: for every group of questions its guesses"""
            asked = {h[0] for group in q_cands for forms in group for _, h in forms}
            top = None
            for s_forms in s_cands:
                if time.time() > until:
                    break
                if q_cands and not all(G.READS[h[0]] & asked for _, h in s_forms):
                    continue                                     # a statement no question can ask about is not guessed here
                can = set().union(*(G.READS[k] for k in told | {h[0] for _, h in s_forms})) if (told or s_forms) else set()
                for lex in lexes:
                    options = [lex]
                    if any(h[0] in loose_kinds or "D" in h[2] or "T" in h[2] for _, h in s_forms):
                        options += self.orders(list(s_forms), lex, some)[:24]      # the words of the statements in those roles
                    elif not s_forms and any(h[0] in loose_kinds for group in q_cands for forms in group for _, h in forms):
                        probe = [f for group in q_cands for forms in group for f in forms if f[1][0] in loose_kinds][:2]
                        options += self.orders(probe, lex, some)[:24]
                    for lex2 in options:
                        base = total(s_forms, lex2)
                        chosen, score = list(s_forms), base
                        for group in q_cands:                    # the best reading of every question, each on its own
                            tops = []
                            for forms in group:
                                if time.time() > until:
                                    break
                                if any(h[0] not in can for _, h in forms):
                                    continue                     # a question can only ask about what statements tell
                                sc = total(list(s_forms) + forms, lex2)
                                if sc > base:
                                    tops.append((sc, forms))
                            for sc, forms in sorted(tops, key=lambda x: -x[0])[:4]:
                                if not self.flat(list(s_forms) + forms, lex2, some):
                                    chosen += forms
                                    score += sc - base
                                    break
                        if score > (top[0] if top else 0):
                            top = (score, chosen, lex2)
            if not top:
                return None
            snap = self.snapshot()                               # with what is like the new forms taken too
            self.take(top[1], top[2])
            self.adopt(every)
            sc = self.score(every)[0]
            added = [(f, h[:3]) for f, h in self.forms.items() if snap[0].get(f, ())[:3] != h[:3]]
            lex = self.lex
            self.restore(snap)
            if (tuple(added), ()) in self.bad:                   # this was tried, and spoiled what was learnt before
                return None
            return (added, sc, lex, None, None) if sc > best and sc >= enough * n_all else None

        s_kinds, q_kinds = list(dark_s.items()), list(dark_q.items())
        size_of = lambda key: sum(w == STAR for w in key)        # noqa: E731
        # 1. sentences of the same build are read alike
        if len(s_kinds) == 2 and size_of(s_kinds[0][0]) == size_of(s_kinds[1][0]):
            (_, one), (_, two) = s_kinds
            s_cands = [forms + [(f, forms[0][1]) for f in two["forms"]] for forms in general(one, False, True)]
            by_size = defaultdict(list)
            for key, kind in q_kinds:
                by_size[size_of(key)].append(kind)
            q_cands = [[forms + [(f, forms[0][1]) for kind in group[1:] for f in kind["forms"]] for forms in general(group[0], True, True)]
                       for group in by_size.values()]
            found = search(s_cands, q_cands, 0.6)
            if found:
                return found
        # 2. freely: every kind of statement and every question on its own
        s_lists = [general(kind, False) for _, kind in s_kinds]
        size = 1
        for x in s_lists:
            size *= len(x)
        if size == 0:
            return None
        one = len(s_lists) > 2 or size > 3000                    # too many together: one kind of statement at a time
        if one:
            s_lists = [min(s_lists, key=len)]
        s_cands = [[f for part in c for f in part] for c in itertools.product(*s_lists)]
        s_cands = [c for c in s_cands if len({f for f, _ in c}) == len(c)]
        q_cands = [general(kind, True) for _, kind in q_kinds]
        return search(s_cands, q_cands, 0.25 if one else 0.6)

    def try_words(self, story, li, stories, best):
        """a word that stands in a place which is not an argument may itself say what is done ("got", "dropped"):
        guess the meaning of such a word in the sentences before the question; it then holds in every form"""
        top, seen = None, set()
        for words, gold in reversed(story[:li]):
            if gold is not None or words in seen or len(seen) >= 8:
                continue
            seen.add(words)
            hit = None
            for form, runs in self.keys(words, False):
                f = self.forms.get(form)
                if f and (hit is None or (len(runs), f[3]) < (len(hit[1]), hit[0][3])):
                    hit = (f, runs)
            if not hit or "O" not in hit[0][2]:
                continue
            (kind, k, roles, _), runs = hit
            s0, e0 = runs[roles.index("O")]
            word = " ".join(words[s0:e0])
            used = {"a": len({f[1] for f in self.forms.values() if G.KINDS[f[0]][2] == "a"}), "o": G.N_ORDER - 1, "m": G.N_MAP - 1}
            for kind2, (need, may, num) in G.STATEMENTS.items():
                if not set(need) <= set(roles) - {"O"} <= set(need) | set(may):
                    continue
                for k2 in ([0] if num == "" else range(min({"a": G.N_ATTR, "o": G.N_ORDER, "m": G.N_MAP}[num], used[num] + 1))):
                    verbs = {word: (kind2, k2)}
                    if self.verbs.get(word, (kind, k)) == (kind2, k2) or ((), tuple(sorted(verbs.items()))) in self.bad or word in self.frozen:
                        continue
                    with self.with_forms([], None, verbs):
                        sc = self.judge([r for s in stories for r in self.run(s)])[0]
                    if sc > (top[0] if top else best):
                        top = (sc, verbs)
        return ([], top[0], None, top[1]) if top else None

    def lexicons(self, group, combo):
        """the single words a guess may need at once, one after another: nothing; forms that look like one word; a pronoun
        in the guessed sentence"""
        yield None
        base = [None]
        fresh = {w: h for w, h in self.maybe_alias.items() if w not in self.lex.alias}
        if fresh:
            lex = self.lex.copy()
            lex.alias.update(fresh)
            base.append(lex)
            yield lex
        pron = set()
        for (words, quest, _), (form, hyp) in zip(group, combo):
            if not quest:
                for f, runs in self.keys(words, quest):
                    if f == form:
                        pron |= {" ".join(words[s:e]) for r, (s, e) in zip(hyp[2], runs) if r == "A1"} - self.lex.pron
        if pron:
            for b in base:
                lex = (b or self.lex).copy()
                lex.pron |= pron
                yield lex

    def orders(self, combo, lex, stories):
        """directions and times for the words a guess uses in those roles: every way to tell them apart"""
        with self.with_forms(combo, lex):
            ds, ts = set(), set()
            for story in stories:
                for words, gold in story:
                    p = self.parse(words, gold is not None)
                    if p:
                        ds |= set(p[2].get("D", ()))
                        ts |= set(p[2].get("T", ()))
        base, out = lex or self.lex, []
        ds, ts = sorted({base.norm(d) for d in ds}), sorted({base.norm(t) for t in ts})
        d_opts = [dict(zip(ds, perm)) for perm in itertools.permutations(G.CODES, len(ds))] if 0 < len(ds) <= 4 else [None]
        t_opts = [dict(zip(ts, perm)) for perm in itertools.permutations(range(len(ts)))] if 1 < len(ts) <= 4 else [None]
        for d in d_opts:
            for t in t_opts:
                if d is None and t is None:
                    continue
                new = base.copy()
                new.dirs.update(d or {})
                new.times.update(t or {})
                out.append(new)
        return out

    def try_group(self, group, story, li, stories, best, until, revise=False):
        """guess some sentences together. First on their own story: only guesses that make more of its answers right go
        on. Then on all stories of the task: the guess with most right answers wins, if it is better than what there is"""
        gold = story[li][1]
        text = {w for words, _ in story for w in words}
        lists = [self.guesses(w, q, again, gold if (w, q) == (story[li][0], True) else None, text) for w, q, again in group]
        size = 1
        for x in lists:
            size *= len(x)
        if size == 0 or size > 2000000:
            return None
        n_all = sum(1 for s in stories for _, g in s if g is not None)
        n_story = sum(1 for _, g in story if g is not None)
        q_at = sum(1 for _, g in story[:li + 1] if g is not None) - 1
        here = self.judge(self.run(story), fixed=True)[0]

        def total(combo, lex):
            with self.with_forms(combo, lex):
                return self.judge([r for s in stories for r in self.run(s)])[0]

        def fits(ans, lex):
            if ans is None:
                return False
            t, v = ans
            al = (lex or self.lex).alias
            return (t == "e" and len(gold) == 1 and al.get(gold[0], gold[0]) == v) or (t == "l" and sorted(v) == sorted(al.get(g, g) for g in gold)) \
                or (t == "s" and len(gold) == 1 and (not set(gold[0].split()) & text or any(gold[0] in c for c in self.said_n.values()))
                    and self.said.get(v, gold[0]) == gold[0]) \
                or (t == "p" and len(v) == len(gold) and all(self.said.get("step " + s, g) == g for s, g in zip(v, gold)))

        told = set()                                             # what the statements of this story tell
        for w2, g2 in story[:li]:
            p2 = self.parse(w2, False) if g2 is None else None
            if p2 and (w2, False, True) not in group:
                told.add(p2[0])
        keep, late, stop = [], [], min(until, time.time() + 60)
        for idx, combo in enumerate(itertools.product(*lists)):
            if idx % 200 == 0 and time.time() > stop:
                break
            if len({f for f, _ in combo}) < len(combo) or (tuple(combo), ()) in self.bad or any(f in self.frozen for f, _ in combo):
                continue
            can = set().union(*(G.READS[k] for k in told | {h[0] for f, h in combo if not f[0]}))
            if any(f[0] and h[0] not in can for f, h in combo):
                continue                                         # a question can only ask about what statements tell
            loose = any(h[0] in ("BEFORE", "PATH", "L_IS", "L_A1", "L_A2", "LINK") or "D" in h[2] or "T" in h[2] for _, h in combo)
            # ... then the answer depends on directions or on the order of times, which may not be known yet
            for lex in self.lexicons(group, combo):
                with self.with_forms(combo, lex):
                    ans = self.run(story, li + 1)[-1][0]                 # first only up to the question itself
                    if not (fits(ans, lex) or loose):
                        continue
                    res = self.run(story)
                    local = self.judge(res, fixed=True)[0]
                if local > here:
                    keep.append((local, idx, combo, lex, loose))
                    if local == n_story and not loose:               # all of this story is right: is the whole task?
                        sc = total(combo, lex)
                        if sc == n_all and sc > best:
                            return list(combo), sc, lex
                    break
                if loose:                                            # cannot be checked before the order is tried
                    late.append((0, idx, combo, lex, loose))
                    break
        keep.sort(key=lambda x: (-x[0], x[1]))
        found = []
        for local, idx, combo, lex, loose in keep[:120]:
            if time.time() > until:
                break
            sc = total(combo, lex)
            if sc > best:
                found.append((sc, len(found), combo, lex))
        for local, idx, combo, lex, loose in ([x for x in keep if x[4]][:15] + late[:60]):
            for lex2 in self.orders(combo, lex, stories):
                if time.time() > until:
                    break
                sc = total(combo, lex2)
                if sc > best:
                    found.append((sc, len(found), combo, lex2))
        top = None
        for sc, _, combo, lex in sorted(found, key=lambda x: (-x[0], x[1]))[:6]:
            if not self.flat(combo, lex, stories):            # always the same answer is not an answer
                top = (sc, combo, lex)
                break
        if not top:
            return None
        if all(again for _, _, again in group) and not revise and top[0] < min(best + 2, n_all):
            return None                                          # one more right answer is not a reason to read a sentence differently
        return list(top[1]), top[0], top[2]

    def words(self, stories, best):
        """single words, each tried and checked: pronouns, directions, times, forms of one word, habits of the ground"""
        start = best

        everything = self.score(self.seen)[0]

        def better(change, undo):
            nonlocal best, everything
            change()
            sc, ev = self.score(stories)[0], self.score(self.seen)[0]
            if sc > best and ev > everything:
                best, everything = sc, ev
                return True
            undo()
            return False

        for attr, options in (("like", ("kind", "told", "most")), ("gave", ("last", "first"))):
            old = getattr(self.lex, attr)
            for o in options:
                if o != getattr(self.lex, attr):
                    if not better(lambda o=o: setattr(self.lex, attr, o), lambda: setattr(self.lex, attr, old)):
                        continue
                    old = o
        roles = defaultdict(set)                                 # what was said in which role
        for story in stories:
            for w, g in story:
                p = self.parse(w, g is not None)
                if p:
                    for r, vals in p[2].items():
                        roles[r] |= set(vals)
        for w in sorted(roles["A1"]):
            if w not in self.lex.pron:
                better(lambda w=w: self.lex.pron.add(w), lambda w=w: self.lex.pron.discard(w))
        for _ in range(2):
            for w in sorted({self.lex.norm(d) for d in roles["D"]}):
                for c in G.CODES:
                    old = self.lex.dirs.get(w)
                    if c != old:
                        better(lambda c=c: self.lex.dirs.__setitem__(w, c),
                               lambda: self.lex.dirs.pop(w, None) if old is None else self.lex.dirs.__setitem__(w, old))
        ts = sorted({self.lex.norm(t) for t in roles["T"]})
        if 1 < len(ts) <= 5:
            old = dict(self.lex.times)
            for perm in itertools.permutations(range(len(ts))):
                new = dict(old, **dict(zip(ts, perm)))
                if better(lambda new=new: setattr(self.lex, "times", new), lambda: setattr(self.lex, "times", old)):
                    old = new
        ds = sorted({self.lex.norm(d) for d in roles["D"]})
        if 0 < len(ds) <= 4:
            old = dict(self.lex.dirs)
            for perm in itertools.permutations(G.CODES, len(ds)):
                new = dict(old, **dict(zip(ds, perm)))
                if better(lambda new=new: setattr(self.lex, "dirs", new), lambda: setattr(self.lex, "dirs", old)):
                    old = new
        for w in sorted(self.lex.alias):                          # a likeness that changes nothing is not kept
            keep, before = self.lex.alias.pop(w), (self.score(stories)[0], self.score(self.seen)[0])
            if before[0] < best or before[1] < everything:
                self.lex.alias[w] = keep
        things = sorted(roles["A1"] | roles["A2"] | roles["A3"])
        if best < sum(1 for s in stories for _, g in s if g is not None) and len(things) <= 40:
            golds = sorted({x for s in stories for _, g in s if g for x in g})
            for a, b in [(a, b) for a in things for b in things + golds if a != b and a not in self.lex.alias and b not in self.lex.alias]:
                better(lambda: self.lex.alias.__setitem__(a, b), lambda: self.lex.alias.pop(a, None))
        said = self.judge([r for s in stories for r in self.run(s)])[1]
        for story in stories:                                    # what was seen to follow from what
            w = G.World(self.lex)
            for words, gold in story:
                p = self.parse(words, gold is not None)
                if not p:
                    continue
                try:
                    if gold is None:
                        w.tell(*p)
                    elif p[0] == "NEXT" and w.ask(*p) is None and len(gold) == 1:
                        v = w.val[p[1][0]].get(w.name(p[2]["A1"][0]))
                        if v is not None:
                            self.lex.kb.setdefault((p[1][0], p[1][1], v), gold[0])
                except (IndexError, KeyError):
                    pass
        del said
        best = self.score(stories)[0]
        return best > start

    # ---------------------------------------------------------------- what was learnt, in words
    def describe(self):
        out = []
        for form, (kind, k, roles, _) in sorted(self.forms.items(), key=lambda x: x[1][3]):
            it = iter(roles)
            text = " ".join(next(it) if w == STAR else next(it) + "+" if w == STARS else w for w in form[1:])
            out.append(f"{'?' if form[0] else ' '} {text:55s} {kind}{'' if G.KINDS[kind][2] == '' else ' ' + str(k)}")
        for w, (kind, k) in sorted(self.verbs.items()):
            out.append(f"  the word '{w}' means {kind}{'' if G.KINDS[kind][2] == '' else ' ' + str(k)}")
        return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a folder of bAbI files (en, hn, shuffled)")
    ap.add_argument("--stories", type=int, default=20, help="stories of every task to learn from")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--budget", type=float, default=600, help="seconds for one task")
    ap.add_argument("--show", action="store_true", help="print the forms that were learnt")
    ap.add_argument("--trace", action="store_true", help="print every guess as it is kept")
    ap.add_argument("--out", default="", help="where to save what was learnt (json)")
    ap.add_argument("--state", default="", help="a file to keep the learner in after every task; a run started again goes on from it")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":                  # sets are walked in the same order in every run
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    only = [int(x) for x in a.tasks.split(",") if x] or list(range(1, 21))
    mind, rows, t0 = Mind(), [], time.time()
    if a.state and os.path.exists(a.state):
        mind = pickle.load(open(a.state, "rb"))
        mind._text, mind._named = {}, {}
        print(f"going on after {len(mind.tasks)} tasks", flush=True)
    mind.trace = a.trace
    for n, k in enumerate(only):
        if n < len(mind.tasks):
            continue
        f = glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0]
        got, total = mind.learn(read(f, a.stories), a.budget)
        print(f"task {k:2d}: learnt from {a.stories} stories, {got}/{total} of their questions right, {len(mind.forms)} forms ({time.time() - t0:.0f} s)", flush=True)
        if a.state:
            pickle.dump(mind, open(a.state + ".new", "wb"))
            os.replace(a.state + ".new", a.state)
    if a.show:
        print(mind.describe())
        print("pronouns:", sorted(mind.lex.pron), "| directions:", mind.lex.dirs, "| times:", mind.lex.times, "| one word:", mind.lex.alias,
              "| said:", mind.said, "| follows:", mind.lex.kb, "|", mind.lex.like, mind.lex.gave)
    right = n = 0
    for k in only:
        f = glob.glob(os.path.join(a.data, f"qa{k}_*test.txt"))[0]
        res = [r for s in read(f) for r in mind.run(s)]
        say = dict(mind.said_all, **{s: c.most_common(1)[0][0] for s, c in mind.said_n.items()})
        r = sum(mind.right(ans, gold, say) for ans, gold in res)
        rows.append((k, r, len(res)))
        right, n = right + r, n + len(res)
        print(f"test {k:2d}: {r}/{len(res)} = {100 * r / len(res):.1f}%", flush=True)
    print(f"all: {right}/{n} = {100 * right / n:.2f}%   ({time.time() - t0:.0f} s)")
    if a.out:
        json.dump({"forms": [[list(f[1:]), f[0], list(h[:3])] for f, h in mind.forms.items()], "pronouns": sorted(mind.lex.pron),
                   "directions": mind.lex.dirs, "times": mind.lex.times, "one_word": mind.lex.alias, "said": mind.said,
                   "follows": [[list(k), v] for k, v in mind.lex.kb.items()], "like": mind.lex.like, "gave": mind.lex.gave,
                   "test": rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
