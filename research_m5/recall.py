#!/usr/bin/env python3
"""
A ground with nothing said of sentences: a story is kept as it was told, and a question is answered by recalling.

records.py gives every kind of sentence a meaning - one of some seventy moves on a state - and looks for the
meanings of all the kinds of sentences of a task together; that search is where its hours go. Here a sentence has no
meaning of its own. The statements of a story stay as records (the kind of sentence, the things and numbers in it).
What is learnt is, for every kind of question apart, how to recall: which kinds of statements it looks at and what
each is to it, and what it asks of them. There are three ways of looking at the records, and nothing else:

    a value:   a statement gives a thing a value (the latest holds), or takes it away      - where a thing is, whose it is
    a link:    a statement puts one thing before another                                   - taller, north of, parent of
    an amount: a statement sets, adds, takes or passes a number                            - how many

and a dozen things to ask of each (the value of a thing; where a chain of values ends; who has this value; how many;
is one before another; the utmost; the amount). Which kind of statement plays which part for a kind of question is
found by trying and checking on the stories, a kind of question at a time: seconds, not hours.

    python recall.py --data new/K --tasks 1,2,3 --show
"""
import argparse
import glob
import itertools
import os
import re
import time
from collections import Counter

import records
import rows


def kind(x):
    return x[0]


def thing(x):
    return bool(x) and x[:1].isalpha() and x[1:].isdigit()


class Task:
    """the stories of a task as records: statements with the kind of sentence as a number, questions apart"""

    def __init__(self, episodes, split=True):
        self.split = self.labels(episodes) if split else {}      # of a kind of sentence: the slot whose word is part of the kind
        self.forms, self.asked, self.stories = {}, {}, []
        for ep in episodes:
            story = []
            for op, quest, things, numbers, kinds, gold in ep:
                f = self.form(op, quest, things, kinds)
                if quest:
                    story.append((self.asked.setdefault(f, len(self.asked)), True, things, numbers, gold))
                else:
                    story.append((self.forms.setdefault(f, len(self.forms)), False, things, numbers, None))
            self.stories.append(story)
        self.names = sorted({x for ep in episodes for row in ep for x in row[2]})
        self.cut = {}                                            # for a kind of question: the stories it is asked in, as far as it is asked
        for q in self.asked.values():
            for story in self.stories:
                last = max((i for i, row in enumerate(story) if row[1] and row[0] == q), default=-1)
                if last >= 0:
                    self.cut.setdefault(q, []).append([row for row in story[:last + 1] if not row[1] or row[0] == q])
        members = {}
        for x in self.names:
            members.setdefault(kind(x), set()).add(x)
        self.few = {k for k, xs in members.items() if len(xs) <= 8}
        self.masks, self.build = {}, {}                          # kinds of sentences of one build: the same but for one word
        for f, n in self.forms.items():
            for key in self.masked(f):
                self.masks.setdefault(key, []).append(n)
        for f, n in self.forms.items():
            keys = [key for key in self.masked(f) if len(self.masks[key]) > 1]
            self.build[n] = max(keys, key=lambda key: len(self.masks[key])) if keys else (f[2], f[0], f[3])

    @staticmethod
    def masked(f):
        words = f[0].split()
        return [(f[2], tuple(words[:i] + ["_"] + words[i + 1:]), f[3]) for i in range(len(words)) if words[i] != "*"]

    def met(self, f):
        """a kind of statement, by its number; one not met yet: by the build of those it differs from in one word"""
        if f in self.forms:
            return self.forms[f]
        keys = [key for key in self.masked(f) if key in self.masks]
        return ("new", max(keys, key=lambda key: len(self.masks[key]))) if keys else -1

    def keys(self, f):
        """the places of a kind of sentence whose thing may be a key: it says of which the sentence speaks (north, the colour)"""
        ks = [k for k in f[2] if k != "#"]
        return [i for i, k in enumerate(ks) if k in self.few] if len(ks) >= 2 else []

    def golds(self, q):
        return [g for story in self.stories for f, quest, _, _, g in story if quest and f == q]

    def own(self, q):
        """the kind of things a kind of question is answered with, if it is always one kind"""
        owns = {kind(x) for g in self.golds(q) for x in g.split(",") if thing(x)}
        return owns.pop() if len(owns) == 1 else None

    @staticmethod
    def labels(episodes):
        """the slots whose words are part of the kind of sentence: a kind of words of which there are few, that stand
        nowhere but in one such place of a statement (taller, shorter; opens, closes) - a sentence with another such
        word is another kind of sentence"""
        where, members, twice = {}, {}, set()
        for ep in episodes:
            for op, quest, things, numbers, kinds, gold in ep:
                ks = [k for k in kinds if k != "#"]
                twice |= {k for k in ks if ks.count(k) > 1}      # (a kind that stands twice in a sentence is what is spoken of)
                for i, (x, k) in enumerate(zip(things, ks)):
                    where.setdefault(k, set()).add((op, quest, i, len(ks)))
                    members.setdefault(k, set()).add(x)
                for x in (gold or "").split(",") if quest else ():
                    if thing(x):
                        where.setdefault(kind(x), set()).add("answer")
        split = {}
        for k, places in where.items():
            said = [p for p in places if p != "answer" and not p[1]]
            if k not in twice and "answer" not in places and len(members[k]) <= 8 and said and all(p[3] >= 2 for p in places):
                for p in places:
                    if p != "answer":
                        split[(p[0], p[1], p[3])] = p[2]
        return split

    def form(self, op, quest, things, kinds):
        ks = tuple(k for k in kinds if k != "#")
        i = self.split.get((op, quest, len(ks)))
        return (op, quest, kinds, things[i] if i is not None else None)


# ---------------------------------------------------------------- the three ways of looking at what was said
def options(sort, nt, nn, consts):
    """what a kind of statement with so many things and numbers may be to a kind of question, for a way of looking"""
    if sort == "value":                                      # a thing gets a thing of the row for its value; or the value another has
        return [("set", i, j) for i in range(nt) for j in range(nt) if i != j] + [("as", i, j) for i in range(nt) for j in range(nt) if i != j] \
            + [("lift", i) for i in range(nt)] + [("clear", i) for i in range(nt)] + [("to", i, c) for i in range(nt) for c in consts]      # now; or the value of
                                                                 # its value (it leaves its holder where the holder is); or none; or a value never named in the row
    if sort == "link":
        return [(name, i, j) for name in ("link", "unlink") for i in range(nt) for j in range(nt) if i != j]
    keys = [(i,) for i in range(nt)] + [(i, j) for i in range(nt) for j in range(nt) if i != j]
    out = [(name, key, n) for name in ("put", "add", "take") for key in keys for n in range(nn)]
    out += [("pass", (i, j) + rest, n) for i in range(nt) for j in range(nt) if i != j for rest in [()] + [(c,) for c in range(nt) if c not in (i, j)] for n in range(nn)]
    return out if nn else [(name, key, None) for name in ("add", "take") for key in keys]         # (with no number said: one)


def keyed(sort, nt, nn, consts, keys):
    """the same, of the thing named in one place of the statement only (a store of its own for every such thing)"""
    out = []
    for k in keys:
        for role in options(sort, nt, nn, consts):
            used = set(role[1]) if sort == "amount" else {x for x in role[1:3] if x.__class__ is int and role[0] != "to"} | {role[1]}
            if k not in used:
                out.append(("at", k, role))
    return out


def said(state, sort, role, things, numbers):
    """a statement, as the part it plays: the state of the story after it"""
    name = role[0]
    if sort == "value":
        if name == "set":
            state[things[role[1]]] = things[role[2]]
        elif name == "as":
            if state.get(things[role[2]]) is not None:
                state[things[role[1]]] = state[things[role[2]]]
        elif name == "lift":
            if state.get(state.get(things[role[1]])) is not None:
                state[things[role[1]]] = state[state[things[role[1]]]]
            else:
                state.pop(things[role[1]], None)
        elif name == "to":
            state[things[role[1]]] = role[2]
        else:
            state.pop(things[role[1]], None)
    elif sort == "link":
        a, b = things[role[1]], things[role[2]]
        if name == "link":
            state.setdefault((a, 1), set()).add(b)
            state.setdefault((b, 0), set()).add(a)
        else:
            state.get((a, 1), set()).discard(b)
            state.get((b, 0), set()).discard(a)
    else:
        n = numbers[role[2]] if role[2] is not None else 1
        if name == "pass":
            a, b = (things[role[1][0]],), (things[role[1][1]],)
            if len(role[1]) == 3:
                a, b = a + (things[role[1][2]],), b + (things[role[1][2]],)
            state[a], state[b] = state.get(a, 0) - n, state.get(b, 0) + n
        else:
            key = tuple(things[i] for i in role[1])
            state[key] = n if name == "put" else state.get(key, 0) + (n if name == "add" else -n)


def chain(state, x):
    seen = [x]
    while state.get(seen[-1]) is not None and state[seen[-1]] not in seen:
        seen.append(state[seen[-1]])
    return seen[1:]


def reach(links, x, back=False):
    out, new = {x}, [x]
    while new:
        for y in links.get((new.pop(), 0 if back else 1), ()):
            if y not in out:
                out.add(y)
                new.append(y)
    return out - {x}


def asks(sort, nt, nn, shape, own, keys=(), later=False):
    """what a kind of question with so many things may ask, for a way of looking: (name, the things of it that it uses);
    and the same asked of the store of the thing in one of its places: (name, the things, the place of the key)"""
    out = asked(sort, list(range(nt)), nn, shape)
    for k in keys:
        out += [ask + (k,) for ask in asked(sort, [i for i in range(nt) if i != k], nn, shape)]
    if not later:
        return out
    more = []                                                # the same asked of another time than now:
    for ask in out[:len(asked(sort, list(range(nt)), nn, shape))]:       # (of the story itself, not of the store of a key)
        free = [i for i in range(nt) if i not in ask[1]]
        if sort != "amount" or ask[0] != "atleast":          # as it was at the time the question names, or before it
            more += [(name, n, ask) for name in ("asof", "until") for n in range(nn)]
        if ask[0] in ("value", "end", "holders"):            # what it was before, at first, before it was what the question names
            more += [("was", ask), ("began", ask)] + [("prior", j, ask) for j in free]
        if sort == "value" and ask[0] in ("value", "end", "holders", "is", "isin", "count"):
            more += [(name, (i, j), ask) for name in ("upto", "ere") for i in free for j in free if i != j]      # when a thing got a value; just before
    return more


WRAPS = ("asof", "until", "was", "began", "prior", "upto", "ere")


def bare(ask):
    """the thing asked, without the time it is asked of"""
    return ask[-1] if ask[0] in WRAPS else ask


def key(ask):
    """the place of the thing of whose store it is asked, if of any"""
    return bare(ask)[2] if len(bare(ask)) > 2 else None


def core(part):
    """the part a statement plays, without the store and the time it plays it at"""
    while part[0] in ("at", "when"):
        part = part[2]
    return part


def recalled(sort, ask, line, things, numbers, own, names):
    """a question of some time: the statements in the order of their times (those of no time first, as told), as far
    as the question says; then what is asked - of the story as it then was, or of how it came to be so"""
    plain, name = bare(ask), ask[0]
    line = sorted(line, key=lambda e: (e[0] is not None, e[0] or 0, e[1]))
    if name in ("asof", "until"):
        t = numbers[ask[1]]
        line = [e for e in line if e[0] is None or (e[0] <= t if name == "asof" else e[0] < t)]
    state, stores, seen, at = {}, {}, [], [None]

    def now():
        return answer(sort, plain, stores.get(things[plain[2]], {}) if len(plain) > 2 else state, things, numbers, own, names)

    def play(entries, watch=None):
        for n, (_, _, part, th, nums) in enumerate(entries):
            try:
                if part[0] == "at":
                    said(stores.setdefault(th[part[1]], {}), sort, part[2], th, nums)
                else:
                    said(state, sort, part, th, nums)
            except (KeyError, IndexError, TypeError):
                pass
            if watch == "past":
                told = now()
                if told is not None and (not seen or seen[-1] != told):
                    seen.append(told)
            elif watch and watch[0] in th and state.get(watch[0]) == watch[1]:
                at[0] = n
    if name in ("upto", "ere"):
        play(line, (things[ask[1][0]], things[ask[1][1]]))
        if at[0] is None:
            return None
        state.clear()
        stores.clear()
        play(line[:at[0] + (name == "upto")])
    elif name in ("was", "began", "prior"):
        play(line, "past")
        if name == "began":
            return seen[0] if seen else None
        if name == "was":
            return seen[-2] if len(seen) > 1 else None
        where = [i for i, v in enumerate(seen) if v == things[ask[1]]]
        return seen[where[-1] - 1] if where and where[-1] > 0 else None
    else:
        play(line)
    return now()


def asked(sort, slots, nn, shape):
    nt = len(slots)
    one, two = [(i,) for i in slots], [(i, j) for i in slots for j in slots if i != j]
    if sort == "value":
        if shape == "t":
            return [(n, a) for n in ("value", "end", "holders", "inside") for a in one] + [("values", a) for a in one] + [("both", a) for a in two]
        if shape == "n":
            return [(n, a) for n in ("count", "countin") for a in one]
        return [(n, a) for n in ("is", "isin", "same") for a in two] + [(n, a) for n in ("any", "known") for a in one]
    if sort == "link":
        if shape == "t":
            return [(n, a) for n in ("after", "before", "next", "prev") for a in one] + [("first", ()), ("last", ())] + ([("firstof", ()), ("lastof", ())] if nt >= 2 else [])
        if shape == "n":
            return [(n, a) for n in ("countafter", "countbefore") for a in one]
        return [("leads", a) for a in two]
    if shape == "n":
        return [("amount", a) for a in one + two] + [("total", a) for a in one] + [("sum", ())] + [("gap", a) for a in two]
    if shape == "t":
        return [(n, a) for n in ("most", "least") for a in [()] + one] + [("more", a) for a in two] + [("less", a) for a in two]
    return [("morethan", a) for a in two] + [("has", a) for a in one + two] + [("atleast", a) for a in one + two if nn]


def answer(sort, ask, state, things, numbers, own, names):
    """what a question says, by the state of the story as it is looked at -> the answer (a string), or None"""
    name, at = ask[:2]
    t = [things[i] for i in at]
    if sort == "value":
        if name == "value":
            return state.get(t[0])
        if name == "values":
            return ",".join(chain(state, t[0])) or None
        if name == "end":
            return (chain(state, t[0]) or [None])[-1]
        if name in ("holders", "count", "any"):
            xs = sorted(x for x, v in state.items() if v == t[0] and (own is None or kind(x) == own or name != "holders"))
            return ",".join(xs) or None if name == "holders" else str(len(xs)) if name == "count" else bool(xs)
        if name in ("inside", "countin"):
            xs = sorted(x for x in state if t[0] in chain(state, x) and (own is None or kind(x) == own or name != "inside"))
            return ",".join(xs) or None if name == "inside" else str(len(xs))
        if name == "both":
            return ",".join(sorted(x for x, v in state.items() if v == state.get(t[0]) and x != t[0])) or None
        if name == "is":
            return state.get(t[0]) == t[1]
        if name == "isin":
            return t[1] in chain(state, t[0])
        if name == "same":
            return None if state.get(t[0]) is None or state.get(t[1]) is None else state[t[0]] == state[t[1]]
        return t[0] in state                                     # known
    if sort == "link":
        if name in ("after", "before", "countafter", "countbefore"):
            xs = sorted(x for x in reach(state, t[0], "before" in name) if own is None or kind(x) == own or name.startswith("count"))
            return str(len(xs)) if name.startswith("count") else ",".join(xs) or None
        if name in ("next", "prev"):
            return ",".join(sorted(x for x in state.get((t[0], 1 if name == "next" else 0), ()) if own is None or kind(x) == own)) or None
        if name == "leads":
            return True if t[1] in reach(state, t[0]) else False if t[0] in reach(state, t[1]) else None
        among = {things[i] for i in range(len(things)) if len(ask) < 3 or i != ask[2]} if name.endswith("of") else {x for x, _ in state}
        ends = [x for x in among if not (reach(state, x, name.startswith("first")) & among)]
        return ends[0] if len(ends) == 1 and len(among) > 1 else None
    if name == "amount":
        return str(state[tuple(t)]) if tuple(t) in state else None
    if name == "has":
        return state.get(tuple(t), 0) > 0
    if name == "atleast":
        return state.get(tuple(t), 0) >= numbers[0]
    if name in ("total", "sum"):
        vs = [v for key, v in state.items() if not t or t[0] in key]
        return str(sum(vs)) if vs else None
    if name == "gap":
        a, b = state.get((t[0],)), state.get((t[1],))
        return None if a is None or b is None else str(abs(a - b))
    if name in ("more", "less", "morethan"):
        a, b = state.get((t[0],)), state.get((t[1],))
        if a is None or b is None or (a == b and name != "morethan"):
            return None
        return a > b if name == "morethan" else t[0] if (a > b) == (name == "more") else t[1]
    held = {key: v for key, v in state.items() if not t or t[0] in key}                      # most, least
    if not held:
        return None
    best = (max if name == "most" else min)(held.values())
    keys = [key for key, v in held.items() if v == best]
    rest = [x for x in keys[0] if not t or x != t[0]] if len(keys) == 1 else []
    return rest[0] if len(rest) == 1 else None


# ---------------------------------------------------------------- learning a kind of question
class Reading:
    """how a kind of question recalls: the way of looking, the part every kind of statement plays, what is asked"""

    def __init__(self, sort, ask, roles, turned, default, right, total):
        self.sort, self.ask, self.roles, self.turned, self.default, self.right, self.total = sort, ask, roles, turned, default, right, total

    def tell(self, task):
        back = {n: f for f, n in task.forms.items()}
        parts = "; ".join(f"{back[f][0]}{' [' + back[f][3] + ']' if back[f][3] else ''} = {' '.join(str(x) for x in role)}" for f, role in sorted((f, r) for f, r in self.roles.items() if f in back))
        return f"{self.sort}: {' '.join(str(x) for x in self.ask)}{' (turned over)' if self.turned else ''}{' (else ' + self.default + ')' if self.default else ''} | {parts}"


def run(task, q, sort, ask, roles, own, stories=None, stop=None):
    """the answers a way of recalling gives to the questions of a kind in the stories -> [(said, gold)]; stop: with more
    answers than that plainly wrong it is not worth going on -> None"""
    out, wrong, plain = [], 0, bare(ask)
    timed = ask is not plain or any(part[0] == "when" for role in roles.values() for part in (role[1:] if role[0] == "two" else (role,)))
    for story in stories if stories is not None else task.cut.get(q, ()):
        state, stores, line = {}, {}, []
        for f, quest, things, numbers, gold in story:
            if quest:
                if f == q:
                    try:
                        if timed:
                            told = recalled(sort, ask, line, things, numbers, own, task.names)
                        else:
                            told = answer(sort, ask, stores.get(things[ask[2]], {}) if len(ask) > 2 else state, things, numbers, own, task.names)
                    except (KeyError, IndexError, TypeError):
                        told = None
                    out.append((told, gold))
                    if stop is not None and told.__class__ is str and told != gold:
                        wrong += 1
                        if wrong > stop:
                            return None
            elif f in roles:
                for part in roles[f][1:] if roles[f][0] == "two" else (roles[f],):
                    try:
                        if timed:
                            when = None
                            if part[0] == "when":
                                when, part = numbers[part[1]], part[2]
                            line.append((when, len(line), part, things, numbers))
                        elif part[0] == "at":
                            said(stores.setdefault(things[part[1]], {}), sort, part[2], things, numbers)
                        else:
                            said(state, sort, part, things, numbers)
                    except (KeyError, IndexError, TypeError):
                        pass
    return out


def grade(pairs):
    """-> (right, turned over?, the answer where nothing is said)"""
    best = (-1, False, None)
    for turned in (False, True):
        told = [(("yes" if x != turned else "no") if x.__class__ is bool else x, g) for x, g in pairs]
        if turned and not any(x.__class__ is bool for x, _ in pairs):
            break
        rest = Counter(g for x, g in told if x is None)
        default = rest.most_common(1)[0][0] if rest else None
        right = sum(x == g for x, g in told if x is not None) + (rest[default] if default else 0)
        if right > best[0]:
            best = (right, turned, default)
    return best


def price(roles, build):
    """what a reading costs: every build of sentence it looks at (the kinds of sentences of one build are the same but
    for one word: to look at one more of them costs nothing, whatever part it plays), a second part of a statement,
    and more for a value from nowhere"""
    parts = {(build[f], r) for f, role in roles.items() for r in (role[1:] if role[0] == "two" else (role,))}
    return 0.25 * len({build[f] for f in roles}) + 0.25 * len({(build[f], role) for f, role in roles.items() if role[0] == "two"}) + 0.75 * sum(core(r)[0] == "to" for _, r in parts)


def worth(score, roles, build):
    """what is explained, less what the reading costs"""
    return score[0] - price(roles, build)


class Search:
    """for one way of looking and one thing asked: the parts of the kinds of statements, found by trying and checking"""

    def __init__(self, task, q, sort, ask, mine, own, limit):
        self.task, self.q, self.sort, self.ask, self.mine, self.own, self.limit = task, q, sort, ask, mine, own, limit
        self.all, self.roles, self.used = len(task.golds(q)), {}, 0
        self.score = grade(run(task, q, sort, ask, {}, own))

    def worth(self):
        return worth(self.score, self.roles, self.task.build)

    def full(self):
        return self.score[0] == self.all

    def better(self, trial):
        if trial == self.roles or self.used > self.limit:
            return False
        build = self.task.build
        pairs = run(self.task, self.q, self.sort, self.ask, trial, self.own, None, self.all - self.worth() - price(trial, build))
        self.used += 1
        got = grade(pairs) if pairs is not None else (-1,)
        if worth(got, trial, build) > self.worth():
            self.roles, self.score = trial, got
            return True
        return False

    def put(self, unit, role):
        return {**{k: v for k, v in self.roles.items() if k not in unit}, **({f: role for f in unit} if role else {})}

    def level(self, units, kick=False):
        """the kinds of statements of a unit all in one part, a unit at a time, as long as it helps; where nothing
        helps alone, two units at once (kick: also when something was found already)"""
        mine = self.mine
        for sweep in range(4):
            moved = False
            for u in units:
                for role in mine[u[0]] + [None]:
                    moved |= self.better(self.put(u, role))
            if not moved and not self.full() and (not self.roles or kick and sweep == 0) and 1 < len(units) <= 8:
                for u, v in itertools.combinations(units, 2):
                    for a in mine[u[0]][:12]:
                        for b in mine[v[0]][:12]:
                            moved |= self.better({**{k: x for k, x in self.roles.items() if k not in u + v}, **{f: a for f in u}, **{f: b for f in v}})
            if not moved or self.full():
                break

    def two(self, units):
        """a statement may say two things at once"""
        for u in units if self.roles and not self.full() else ():
            now = self.roles.get(u[0])
            if now and now[0] != "two" and all(self.roles.get(f) == now for f in u):
                for role in self.mine[u[0]]:
                    if role != now:
                        self.better(self.put(u, ("two", now, role)))

    def reading(self):
        return Reading(self.sort, self.ask, self.roles, self.score[1], self.score[2], self.score[0], self.all)


def learn(task, q, limit=5000, keep=3, deep=True):
    """the best way of recalling for a kind of question. Every way of looking and every thing to ask of it is tried
    coarsely - the kinds of statements with the same kinds of things all in one part; the few of every way of looking
    that explain most are then settled finely: the builds, every kind of statement by itself, two parts at once. First what may be asked
    of the story as it is now; if that does not explain every question (deep), what may be asked of another time
    -> Reading"""
    form = next(f for f, n in task.asked.items() if n == q)
    golds, own = task.golds(q), task.own(q)
    nt = sum(k != "#" for k in form[2])
    nn = len(form[2]) - nt
    shape = "b" if all(g in ("yes", "no") for g in golds) else "n" if all(g.isdigit() for g in golds) else "t"
    consts = sorted({x for g in golds for x in g.split(",") if thing(x)})[:6] if shape != "n" else []
    alike, builds = {}, {}
    for f, n in task.forms.items():
        alike.setdefault(f[2], []).append(n)
        builds.setdefault(task.build[n], []).append(n)
    best = None
    for later in (False, True) if deep else (False,):
        found = []
        for sort in ("value", "link", "amount"):
            plain, bykey = parts(task, sort, consts if sort == "value" else ())
            for ask in asks(sort, nt, nn, shape, own, task.keys(form), later):
                s = Search(task, q, sort, ask, bykey if key(ask) is not None else plain, own, limit)
                s.level([u for u in (tuple(f for f in fs if f in s.mine) for fs in alike.values()) if u])
                if s.full():
                    return s.reading()
                found.append(s)                                  # (one for which nothing was found coarsely may still be settled finely)
        found.sort(key=lambda s: -s.worth())
        for s in [s for sort in ("value", "link", "amount") for s in [x for x in found if x.sort == sort][:keep]]:
            units = [u for u in (tuple(f for f in fs if f in s.mine) for fs in builds.values()) if u]
            s.level(units, True)
            s.level([(f,) for f in s.mine], True)
            s.two(units)
            if s.roles and (best is None or s.worth() > best.worth()):
                best = s
            if s.full():
                return s.reading()
    return best.reading() if best else None


def parts(task, sort, consts=()):
    """for a way of looking: what every kind of statement may be to a kind of question -> plainly, and by a key"""
    plain, bykey = {}, {}
    for f, n in task.forms.items():
        nt = sum(k != "#" for k in f[2])
        plain[n] = options(sort, nt, len(f[2]) - nt, consts)
        bykey[n] = keyed(sort, nt, len(f[2]) - nt, consts, task.keys(f))
        for i in range(len(f[2]) - nt):                          # (a number of the statement may be its time: then it is no amount)
            plain[n] = plain[n] + [("when", i, role) for role in options(sort, nt, 0, consts)]
    return {f: v for f, v in plain.items() if v}, {f: v for f, v in bykey.items() if v}


def asking(task, q, sort, roles, own, deep=False):
    """for parts already given: the best thing a kind of question may ask of them -> (right, ask, turned over, default)"""
    form = next(f for f, n in task.asked.items() if n == q)
    golds = task.golds(q)
    nt = sum(k != "#" for k in form[2])
    shape = "b" if all(g in ("yes", "no") for g in golds) else "n" if all(g.isdigit() for g in golds) else "t"
    best = (-1, None, False, None)
    timed = any(part[0] == "when" for role in roles.values() for part in (role[1:] if role[0] == "two" else (role,)))
    for ask in asks(sort, nt, len(form[2]) - nt, shape, own, task.keys(form)) + (asks(sort, nt, len(form[2]) - nt, shape, own, task.keys(form), True) if deep or timed else []):
        got = grade(run(task, q, sort, ask, roles, own))
        if got[0] > best[0]:
            best = (got[0], ask) + got[1:]
    return best


class Learner:
    """every kind of question learns to recall by itself; then the kinds of questions settle on one reading of the
    statements for a way of looking - they ask of one world, and what one kind of question cannot tell from its few
    answers another can"""

    def __init__(self, episodes, split=True, before=None, deep=True):
        self.task = task = Task(episodes, split)
        self.count = {q: len(task.golds(q)) for q in task.asked.values()}
        self.usual = {q: Counter(task.golds(q)).most_common(1)[0][0] for q in task.asked.values()}
        self.own = {q: task.own(q) for q in task.asked.values()}
        self.readings = {}
        same = before and all(task.forms.get(f) == n for f, n in before.task.forms.items()) and all(task.asked.get(f) == n for f, n in before.task.asked.items())
        for q in task.asked.values():                            # (learnt before from fewer stories: what still explains every
            r = before.readings.get(q) if same else None         # question of its kind is kept as it is)
            r = self.fits(q, r.sort, {f: v for f, v in r.roles.items() if f.__class__ is int}, r.ask[0] in WRAPS) if r else None
            self.readings[q] = r if r and r.right == r.total else learn(task, q, deep=deep)
        self.together()
        for r in self.readings.values():                         # a kind of sentence not met yet, the same but for one word as
            for key, fs in task.masks.items() if r else ():      # those that all play one part to this kind of question: it plays it too
                parts = {r.roles.get(f) for f in fs}
                if len(parts) == 1 and None not in parts:
                    r.roles[("new", key)] = parts.pop()

    def fits(self, q, sort, roles, deep=None):
        """a kind of question with the parts given: the best it may ask of them (of another time too, if it asks so
        already or the parts have times) -> Reading"""
        now = self.readings.get(q)
        got = asking(self.task, q, sort, roles, self.own[q], (bool(now) and now.ask[0] in WRAPS) if deep is None else deep)
        return Reading(sort, got[1], roles, got[2], got[3], got[0], self.count[q]) if got[1] else None

    def agreed(self, new, saved):
        """may kinds of questions change their readings for one in common? Those explained in full stay so; the others
        may lose an answer for a part of a reading they need no more"""
        old = self.readings
        return all(r and (r.right >= old[q].right or old[q].right < old[q].total) for q, r in new.items()) \
            and sum(old[q].right - r.right for q, r in new.items()) <= saved / 0.25

    def free(self, sort, roles, pool):
        """the kinds of statements whose part these kinds of questions cannot tell: their answers are the same without it"""
        task, readings = self.task, self.readings
        base = {q: run(task, q, sort, readings[q].ask, roles, self.own[q]) for q in pool}
        return [f for f in task.forms.values() if f not in roles or all(
            run(task, q, sort, readings[q].ask, {k: v for k, v in roles.items() if k != f}, self.own[q]) == base[q] for q in pool)]

    def adopt(self, pool, q, free):
        """a kind of question of another reading taken into this one: by the parts as they are, or with the parts
        settled that the questions of this reading cannot tell -> was it taken"""
        task, readings, old = self.task, self.readings, self.readings[q]
        sort, roles, members = pool
        saved = price({f: v for f, v in old.roles.items() if f.__class__ is int}, task.build)
        new = self.fits(q, sort, roles)
        if new and self.agreed({q: new}, saved):
            readings[q] = new
            pool[2] = members + [q]
            return True
        form = next(f for f, n in task.asked.items() if n == q)
        golds = task.golds(q)
        nt = sum(k != "#" for k in form[2])
        shape = "b" if all(g in ("yes", "no") for g in golds) else "n" if all(g.isdigit() for g in golds) else "t"
        plain, bykey = parts(task, sort)
        for ask in asks(sort, nt, len(form[2]) - nt, shape, self.own[q], task.keys(form))[:16]:
            mine = bykey if key(ask) is not None else plain
            trial, best = roles, grade(run(task, q, sort, ask, roles, self.own[q]))[0]
            for sweep in range(2):
                moved = False
                for f in free:
                    for role in mine.get(f, []) + [None]:
                        now = {**trial, f: role} if role else {k: v for k, v in trial.items() if k != f}
                        if now != trial:
                            got = grade(run(task, q, sort, ask, now, self.own[q]))[0]
                            if got > best:
                                trial, best, moved = now, got, True
                if not moved or best == self.count[q]:
                    break
            if best >= old.right and trial is not roles:
                new = {p: self.fits(p, sort, trial) for p in members + [q]}
                if self.agreed(new, saved):
                    readings.update(new)
                    pool[1], pool[2] = trial, members + [q]
                    return True
        return False

    def settle(self, sort, roles, pool):
        """the parts of a reading settled on all the questions that look by it at once -> the parts"""
        task, readings = self.task, self.readings
        plain, bykey = parts(task, sort)
        mine = bykey if any(key(readings[q].ask) is not None for q in pool) else plain
        full = all(readings[q].right == readings[q].total for q in pool)

        def total(trial):
            return sum(grade(run(task, q, sort, readings[q].ask, trial, self.own[q]))[0] for q in pool) - price(trial, task.build)
        best = total(roles)
        for sweep in range(2):
            moved = False
            for f in list(roles) if full else mine:              # (while a question is left unexplained: the kinds of statements not looked at too)
                now = roles.get(f, ("",))
                if now[0] == "two" or core(now)[0] == "to":
                    continue
                for role in mine.get(f, []) + [None]:
                    trial = {**roles, f: role} if role else {k: v for k, v in roles.items() if k != f}
                    if trial != roles:
                        got = total(trial)
                        if got > best:
                            roles, best, moved = trial, got, True
            new = {q: self.fits(q, sort, roles) for q in pool}
            if all(new[q] and new[q].right >= readings[q].right for q in pool):
                readings.update(new)
            else:
                roles = readings[pool[0]].roles
                break
            if not moved:
                break
        return roles

    def together(self):
        task, readings = self.task, self.readings
        pools = []                                               # [the way of looking, the parts, the kinds of questions that look so]
        for sort in ("value", "link", "amount"):
            for q in sorted((q for q, r in readings.items() if r and r.sort == sort), key=lambda q: (readings[q].right < readings[q].total, -readings[q].right)):
                for pool in [p for p in pools if p[0] == sort]:  # with the others of its way of looking, if their parts agree:
                    mine, best = readings[q].roles, None             # where they differ, the part of the one or of the other
                    apart = [f for f in mine if f in pool[1] and mine[f] != pool[1][f]][:5]
                    for pick in itertools.product((False, True), repeat=len(apart)):
                        trial = {**mine, **pool[1], **{f: mine[f] for f, own in zip(apart, pick) if own}}
                        new = {p: self.fits(p, sort, trial) for p in pool[2] + [q]}
                        got = sum(r.right for r in new.values() if r)
                        if (best is None or got > best[0]) and self.agreed(new, price(mine, task.build) + price(pool[1], task.build) - price(trial, task.build)):
                            best = (got, trial, new)
                    if best:
                        readings.update(best[2])
                        pool[1], pool[2] = best[1], pool[2] + [q]
                        break
                else:
                    pools.append([sort, readings[q].roles, [q]])
        for pool in pools:
            if len(pool[2]) > 1:
                pool[1] = self.settle(*pool)
        pools.sort(key=lambda p: -sum(readings[q].right for q in p[2]))
        for i, big in enumerate(pools):                          # a reading that explains a kind of question of another as well takes it:
            free = None                                          # as it is, or with the parts settled that its own questions cannot tell
            for small in pools[i + 1:] if big[2] else ():
                for q in list(small[2]):
                    free = self.free(*big) if free is None else free
                    if self.adopt(big, q, free):
                        small[2].remove(q)
                        free = None
        for q, r in readings.items():                            # a kind of question nothing was found for: by a reading of the others
            for sort, roles, pool in pools if r is None else ():
                new = self.fits(q, sort, roles) if pool else None
                if new and new.right > max(Counter(task.golds(q)).most_common(1)[0][1], readings[q].right if readings[q] else 0):
                    readings[q] = new

    def cost(self):
        """what the readings cost, one that several kinds of questions share counted once"""
        return sum({id(r.roles): price({f: v for f, v in r.roles.items() if f.__class__ is int}, self.task.build) for r in self.readings.values() if r}.values())

    def explained(self):
        return sum(r.right for r in self.readings.values() if r), sum(self.count.values())

    def test(self, episodes):
        """-> right, all; and of the kinds of questions it has explained in full: right, all"""
        right = total = sure_right = sure_total = 0
        task, self.tally = self.task, {}
        for ep in episodes:
            story = []
            for op, quest, things, numbers, kinds, gold in ep:
                f = task.form(op, quest, things, kinds)
                story.append((task.asked.get(f, -1) if quest else task.met(f), quest, things, numbers, gold))
            for q, r in self.readings.items():
                for said_, gold in run(task, q, r.sort, r.ask, r.roles, self.own[q], [story]) if r else ():
                    told = ("yes" if said_ != r.turned else "no") if said_.__class__ is bool else said_
                    told = r.default if told is None else told
                    right += told == gold
                    total += 1
                    mine = self.tally.setdefault(q, [0, 0])
                    mine[0], mine[1] = mine[0] + (told == gold), mine[1] + 1
                    if r.right == r.total and r.total >= 3:      # (a kind of question met once or twice is not known)
                        sure_right += told == gold
                        sure_total += 1
                if not r:
                    hit = [g == self.usual[q] for f2, quest, _, _, g in story if quest and f2 == q]
                    right, total = right + sum(hit), total + len(hit)
            total += sum(1 for f2, quest, _, _, _ in story if quest and f2 == -1)
        return right, total, sure_right, sure_total


def study(reader, raw, stories=20, ask=20, split=True):
    """learn from the first stories; then, five stories at a time, check on the next ones and, where an answer is
    wrong, learn from them too (they are asked for: at most so many) -> the learner, how many it asked for, has it
    settled (twice in a row nothing was wrong, before it had asked for all it might)"""
    eps = reader.rows(raw)
    used, clean = stories, 0
    m = Learner(eps[:used], split)
    while used - stories < ask and len(eps) > used:
        right, total, _, _ = m.test(eps[used:used + 5])
        used += 5
        clean = clean + 1 if right == total and m.explained()[0] == m.explained()[1] else 0
        if clean == 2:                                           # (twice in a row nothing was wrong: enough)
            break
        if not clean:
            m = Learner(eps[:used], split, m)
    return m, used - stories, clean == 2 and used - stories < ask


class Records:
    """stories that are records already (no words: the name of an operation and its things), as a reader of them"""
    guessed = ()

    @staticmethod
    def rows(raw):
        return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--records", action="store_true", help="the stories are records, not sentences")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--stories", type=int, default=20)
    ap.add_argument("--ask", type=int, default=20, help="how many more stories it may ask the answers of")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    files = glob.glob(os.path.join(a.data, "qa*_train.txt"))
    tasks = [int(x) for x in a.tasks.split(",") if x] or sorted(int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1)) for f in files)
    for k in tasks:
        t0 = time.time()
        path = glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0]
        raw = records.read(path, a.stories + a.ask) if a.records else rows.stories(path, a.stories + a.ask)
        tried, plain = [], None                                  # the stories as they are, then with one guess about their words (which
        for deep in (False, True):                               # of them are things; whether a word is part of the kind of sentence):
            for reader in [Records] if a.records else rows.guesses(raw[:a.stories]):     # each weighed by what it explains, less what its
                for split in (False, True):                      # readings cost and half a question for every kind of question it makes;
                    m = Learner(reader.rows(raw[:a.stories]), split, deep=deep)      # first quickly (of the story as it is now), then,
                    right, total = m.explained()                                     # if nothing explains all, in full
                    plain = plain or len(m.readings)
                    tried.append((right - 0.5 * len(m.readings) - m.cost() - 3 * bool(reader.guessed), right == total, reader, split))    # (a guess is paid for too)
                    if right == total or not m.task.split:
                        break
                if tried[-1][1] and len(m.readings) <= plain:    # (it explains all and makes no more kinds of questions than the stories as they are)
                    break
            best = max(tried, key=lambda t: t[0])
            if best[1]:
                break
        reader, split = best[2], best[3]
        m, asked, settled = study(reader, raw, a.stories, a.ask, split)
        right, total = m.explained()
        full = sum(bool(r) and r.right == r.total and r.total >= 3 for r in m.readings.values())
        print(f"task {k:2d}: learnt from {a.stories} stories (asked for {asked} more), {right}/{total} of their questions right, {full} of {len(m.readings)} kinds of questions explained in full "
              f"({time.time() - t0:.1f} s)" + (" - " + "; ".join(reader.guessed) if reader.guessed else ""), flush=True)
        if a.show:
            back = {n: f for f, n in m.task.asked.items()}
            for q, r in sorted(m.readings.items()):
                print(f"  ? {back[q][0]}{' [' + back[q][3] + ']' if back[q][3] else ''}: " + (f"{r.right}/{r.total} {r.tell(m.task)}" if r else "-")[:300], flush=True)
        path = glob.glob(os.path.join(a.data, f"qa{k}_*test.txt"))[0]
        tests = reader.rows(records.read(path) if a.records else rows.stories(path))
        r, t, sr, st = m.test(tests)
        print(f"test {k:2d}: {r}/{t} = {100 * r / max(1, t):.1f}%; sure of {full} of {len(m.readings)} kinds of questions, {sr}/{st} right on them"
              + ("; settled" if settled and full == len(m.readings) else ""), flush=True)
        if a.show:
            back = {n: f for f, n in m.task.asked.items()}
            for q, (x, y) in sorted(m.tally.items()):
                if x < y:
                    print(f"      on the test {x}/{y} ? {back[q][0]}{' [' + back[q][3] + ']' if back[q][3] else ''}", flush=True)


if __name__ == "__main__":
    main()
