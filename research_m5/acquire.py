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
    if a[0] == b[0] and a[-1] == b[-1] and abs(len(a) - len(b)) <= 1 and len(a) >= 4:
        return 5
    return 0


class Mind:
    def __init__(self):
        self.lex = G.Lexicon()
        self.forms = {}                    # form -> (scheme, number, the role of every argument place)
        self.single, self.multi = set(), set()
        self.said = {}                     # a symbol of the ground -> how it is said
        self.said_n = defaultdict(Counter)
        self.maybe_alias = {}              # forms that look like one word: tried when a guess needs them
        self.sort = {}                     # a word -> the sort of words it can be replaced with
        self.apart = set()                 # pairs of words seen in one place
        self.seen = []                     # the stories learnt from
        self.tasks = []                    # the same stories, task by task
        self._keys, self._parse, self._by_form = {}, {}, defaultdict(set)
        self.trace = False

    # ---------------------------------------------------------------- places that vary
    def sort_of(self, filler):
        """the sort of a word or of several words standing as one (a tuple)"""
        filler = filler[0] if len(filler) == 1 else tuple(filler)
        return self.sort.get(filler, filler)

    def find_places(self, stories):
        golds = {g for story in stories for _, gold in story for g in (gold or ())}
        sents = {(words, gold is not None) for story in stories for words, gold in story}
        apart = self.apart                                     # pairs seen in one place: they are of one sort, and are different things

        def sorts():
            self.sort = {x: x for pair in apart for x in pair}

            def top(w):
                while self.sort[w] != w:
                    w = self.sort[w]
                return w
            for a, b in apart:
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
            if sum(m > 0 for m in mark) > 4:
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
        if best:
            (kind, k, roles, _), runs = best
            a = {}
            for r, (s, e) in zip(roles, runs):
                if r != "O":                                  # a place that varies but does not matter
                    a.setdefault(r, []).append(" ".join(words[s:e]))
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
            if len(gold) > 1:
                can = ("LIST", "PATH")
            elif set(gold[0].split()) <= set(text):                 # the answer is named in the story: a thing is asked
                can = ("VAL", "BEFORE", "P_WHAT", "P_FROM", "P_TO", "NEXT", "L_A1", "L_A2", "LIST")
            else:                                                   # it is not: yes or no, a number, a step
                can = ("IS", "MORE", "L_IS", "COUNT", "LIST", "PATH")
            kinds = {k: v for k, v in kinds.items() if k in can}
        first, out = [], []
        for form, runs in self.keys(words, quest):
            if (form in self.forms and not again) or not runs:
                continue
            # by likeness: a form already known with one more place (a time, a word that does not matter)
            stars = [i for i, w in enumerate(form) if w in (STAR, STARS)]
            for at in stars:                                        # ... or with two places where it has one ("red square" for "triangle")
                if form[at] == STARS:
                    known = self.forms.get(form[:at] + (STAR,) + form[at + 1:])
                    if known and known[0] in kinds:
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
                        else:
                            ks = range(min(room[num], len(used[num]) + 1))        # a new number only after the old ones
                        out += [(form, (kind, k, roles)) for k in ks]
        out.sort(key=lambda g: (len(g[1][2]), len(g[1][2]) - len(set(g[1][2]))))      # the simpler guess first
        self._like = set(first)
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
        for words, gold in story[:upto]:
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

    def judge(self, results, learn=False):
        """how many answers are right, with the symbols of the ground said in the way that fits best"""
        count = defaultdict(Counter)
        for n in (self.said_n, ):
            for s, c in n.items():
                count[s].update(c)
        now = defaultdict(Counter)
        for ans, gold in results:
            if ans and ans[0] == "s" and len(gold) == 1:
                now[ans[1]][gold[0]] += 1
            elif ans and ans[0] == "p" and len(ans[1]) == len(gold):
                for step, g in zip(ans[1], gold):
                    now["step " + step][g] += 1
        for s, c in now.items():
            count[s].update(c)
        said, taken = {}, set()
        for s in sorted(count, key=lambda s: -max(count[s].values())):
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

    def score(self, stories):
        res = [r for s in stories for r in self.run(s)]
        return self.judge(res)[0], len(res)

    def with_forms(self, new, lex=None):
        """a context: the forms with some added or replaced, and maybe other single words"""
        mind = self

        class Trial:
            def __enter__(self):
                self.old = {f: mind.forms.get(f) for f, _ in new}
                self.lex = mind.lex
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
                self.forget()
        return Trial()

    # ---------------------------------------------------------------- learning
    def snapshot(self):
        return dict(self.forms), self.lex.copy(), dict(self.said)

    def restore(self, snap):
        self.forms, self.lex, self.said = dict(snap[0]), snap[1], dict(snap[2])
        self._parse.clear()

    def take(self, new, lex):
        for f, h in new:
            self.forms[f] = h + (self.forms[f][3] if f in self.forms else len(self.forms),)
        if lex is not None:
            self.lex = lex
        self._parse.clear()

    def learn(self, stories, budget=600):
        """one more task: its stories with answers"""
        t0 = time.time()
        before = [self.score(t)[0] for t in self.tasks]            # what the earlier tasks give now: it must not get worse
        self.seen += stories
        self.tasks.append(stories)
        self.find_places(self.seen)
        self.adopt(stories)
        best = self.score(stories)[0]
        total = sum(1 for s in stories for _, g in s if g is not None)
        stuck = set()
        while best < total and time.time() - t0 < budget:
            step = self.next_step(stories, best, stuck, t0 + budget)
            if step is None:
                if not self.words(stories, best):
                    break
                best = self.score(stories)[0]
                stuck.clear()
                continue
            new, sc, lex, unit = step
            snap = self.snapshot()
            self.take(new, lex)
            if self.trace:
                print(f"   + {sc}/{total} ({time.time() - t0:.0f} s): " + "; ".join(" ".join(map(str, f[1:])) + " = " + str(h) for f, h in new), flush=True)
            if any(self.score(t)[0] < b for t, b in zip(self.tasks, before)):
                # the new understanding spoils an earlier task: read that task again in its light, or give the guess up
                if not self.repair(before, t0 + budget) or self.score(stories)[0] <= best:
                    self.restore(snap)
                    stuck.add(unit)
                    continue
            best = self.score(stories)[0]
            self.judge([r for s in stories for r in self.run(s)], learn=True)
        self.words(stories, best)
        best = self.score(stories)[0]
        res = [r for s in stories for r in self.run(s)]
        _, said = self.judge(res, learn=True)
        for ans, gold in res:                                # how the symbols are said stays for the next tasks
            if ans and ans[0] == "s" and len(gold) == 1 and said.get(ans[1]) == gold[0]:
                self.said_n[ans[1]][gold[0]] += 1
            elif ans and ans[0] == "p" and self.right(ans, gold, said):
                for s, g in zip(ans[1], gold):
                    self.said_n["step " + s][g] += 1
        return best, total

    def adopt(self, stories):
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
                if gold is not None or self.parse(words, False) is not None:
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
                    self.take(step[0], step[2])
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
                for group in sets:
                    if time.time() > until:
                        return None
                    found = self.try_group(group, story, li, stories, best, until, revise)
                    if found:
                        return found + ((id(story), li),)
                stuck.add((id(story), li))
        return None

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
        ds, ts = sorted(ds), sorted(ts)
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
        here = self.judge(self.run(story))[0]

        def total(combo, lex):
            with self.with_forms(combo, lex):
                return self.judge([r for s in stories for r in self.run(s)])[0]

        def fits(ans, lex):
            if ans is None:
                return False
            t, v = ans
            al = (lex or self.lex).alias
            return (t == "e" and len(gold) == 1 and al.get(gold[0], gold[0]) == v) or (t == "l" and sorted(v) == sorted(al.get(g, g) for g in gold)) \
                or (t == "s" and len(gold) == 1 and not set(gold[0].split()) & text and self.said.get(v, gold[0]) == gold[0]) \
                or (t == "p" and len(v) == len(gold) and all(self.said.get("step " + s, g) == g for s, g in zip(v, gold)))

        keep, late, stop = [], [], min(until, time.time() + 60)
        for idx, combo in enumerate(itertools.product(*lists)):
            if idx % 200 == 0 and time.time() > stop:
                break
            if len({f for f, _ in combo}) < len(combo):
                continue
            loose = any(h[0] in ("BEFORE", "PATH", "L_IS", "L_A1", "L_A2", "LINK") or "D" in h[2] or "T" in h[2] for _, h in combo)
            # ... then the answer depends on directions or on the order of times, which may not be known yet
            for lex in self.lexicons(group, combo):
                with self.with_forms(combo, lex):
                    ans = self.run(story, li + 1)[-1][0]                 # first only up to the question itself
                    if not (fits(ans, lex) or loose):
                        continue
                    res = self.run(story)
                    local = self.judge(res)[0]
                if local > here:
                    keep.append((local, idx, combo, lex, loose))
                    if local == n_story and not loose:               # all of this story is right: is the whole task?
                        sc = total(combo, lex)
                        if sc == n_all:
                            return list(combo), sc, lex
                    break
                if loose:                                            # cannot be checked before the order is tried
                    late.append((0, idx, combo, lex, loose))
                    break
        keep.sort(key=lambda x: (-x[0], x[1]))
        top = None
        for local, idx, combo, lex, loose in keep[:120]:
            if time.time() > until:
                break
            sc = total(combo, lex)
            if sc > (top[0] if top else best):
                top = (sc, combo, lex)
        for local, idx, combo, lex, loose in ([x for x in keep if x[4]][:15] + late[:60]):
            for lex2 in self.orders(combo, lex, stories):
                if time.time() > until:
                    break
                sc = total(combo, lex2)
                if sc > (top[0] if top else best):
                    top = (sc, combo, lex2)
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
            for w in sorted(roles["D"]):
                for c in G.CODES:
                    old = self.lex.dirs.get(w)
                    if c != old:
                        better(lambda c=c: self.lex.dirs.__setitem__(w, c),
                               lambda: self.lex.dirs.pop(w, None) if old is None else self.lex.dirs.__setitem__(w, old))
        ts = sorted(roles["T"])
        if 1 < len(ts) <= 5:
            old = dict(self.lex.times)
            for perm in itertools.permutations(range(len(ts))):
                new = dict(old, **dict(zip(ts, perm)))
                if better(lambda new=new: setattr(self.lex, "times", new), lambda: setattr(self.lex, "times", old)):
                    old = new
        ds = sorted(roles["D"])
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
    a = ap.parse_args()
    only = [int(x) for x in a.tasks.split(",") if x] or list(range(1, 21))
    mind, rows, t0 = Mind(), [], time.time()
    mind.trace = a.trace
    for k in only:
        f = glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0]
        got, total = mind.learn(read(f, a.stories), a.budget)
        print(f"task {k:2d}: learnt from {a.stories} stories, {got}/{total} of their questions right, {len(mind.forms)} forms ({time.time() - t0:.0f} s)", flush=True)
    if a.show:
        print(mind.describe())
        print("pronouns:", sorted(mind.lex.pron), "| directions:", mind.lex.dirs, "| times:", mind.lex.times, "| one word:", mind.lex.alias,
              "| said:", mind.said, "| follows:", mind.lex.kb, "|", mind.lex.like, mind.lex.gave)
    right = n = 0
    for k in only:
        f = glob.glob(os.path.join(a.data, f"qa{k}_*test.txt"))[0]
        res = [r for s in read(f) for r in mind.run(s)]
        r = sum(mind.right(ans, gold, mind.said) for ans, gold in res)
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
