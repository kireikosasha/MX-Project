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

import rows


def kind(x):
    return x[0]


def thing(x):
    return bool(x) and x[:1].isalpha() and x[1:].isdigit()


class Task:
    """the stories of a task as records: statements with the kind of sentence as a number, questions apart"""

    def __init__(self, episodes):
        self.split = self.labels(episodes)                       # of a kind of sentence: the slot whose word is part of the kind
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

    @staticmethod
    def labels(episodes):
        """the slots whose words are part of the kind of sentence: a kind of words of which there are few, that stand
        nowhere but in one such place of a statement (taller, shorter; opens, closes) - a sentence with another such
        word is another kind of sentence"""
        where, members = {}, {}
        for ep in episodes:
            for op, quest, things, numbers, kinds, gold in ep:
                ks = [k for k in kinds if k != "#"]
                for i, (x, k) in enumerate(zip(things, ks)):
                    where.setdefault(k, set()).add((op, quest, i, len(ks)))
                    members.setdefault(k, set()).add(x)
                for x in (gold or "").split(",") if quest else ():
                    if thing(x):
                        where.setdefault(kind(x), set()).add("answer")
        split = {}
        for k, places in where.items():
            said = [p for p in places if p != "answer" and not p[1]]
            if len(members[k]) <= 8 and said and len({(p[0], p[2]) for p in said}) <= 2 and all(p == "answer" or p[3] >= 2 for p in places):
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
        return [("link", i, j) for i in range(nt) for j in range(nt) if i != j]
    keys = [(i,) for i in range(nt)] + [(i, j) for i in range(nt) for j in range(nt) if i != j]
    out = [(name, key, n) for name in ("put", "add", "take") for key in keys for n in range(nn)]
    out += [("pass", (i, j) + rest, n) for i in range(nt) for j in range(nt) if i != j for rest in [()] + [(c,) for c in range(nt) if c not in (i, j)] for n in range(nn)]
    return out if nn else [(name, key, None) for name in ("add", "take") for key in keys]         # (with no number said: one)


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
        state.add((things[role[1]], things[role[2]]))
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
    out, new = set(), {x}
    while new:
        out |= new
        new = {(a if back else b) for a, b in links if (b if back else a) in new} - out
    return out - {x}


def asks(sort, nt, nn, shape, own):
    """what a kind of question with so many things may ask, for a way of looking: (name, the things of it that it uses)"""
    one, two = [(i,) for i in range(nt)], [(i, j) for i in range(nt) for j in range(nt) if i != j]
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
    name, at = ask
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
            xs = sorted(reach(state, t[0], "before" in name))
            return str(len(xs)) if name.startswith("count") else ",".join(xs) or None
        if name in ("next", "prev"):
            return ",".join(sorted((b if name == "next" else a) for a, b in state if (a if name == "next" else b) == t[0])) or None
        if name == "leads":
            return True if t[1] in reach(state, t[0]) else False if t[0] in reach(state, t[1]) else None
        among = set(things) if name.endswith("of") else {x for pair in state for x in pair}
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
        parts = "; ".join(f"{back[f][0]}{' [' + back[f][3] + ']' if back[f][3] else ''} = {' '.join(str(x) for x in role)}" for f, role in sorted(self.roles.items()))
        return f"{self.sort}: {self.ask[0]} {self.ask[1]}{' (turned over)' if self.turned else ''}{' (else ' + self.default + ')' if self.default else ''} | {parts}"


def run(task, q, sort, ask, roles, own, stories=None):
    """the answers a way of recalling gives to the questions of a kind in the stories -> [(said, gold)]"""
    out = []
    for story in stories if stories is not None else task.stories:
        state = set() if sort == "link" else {}
        for f, quest, things, numbers, gold in story:
            if quest:
                if f == q:
                    try:
                        out.append((answer(sort, ask, state, things, numbers, own, task.names), gold))
                    except (KeyError, IndexError, TypeError):
                        out.append((None, gold))
            elif f in roles:
                try:
                    said(state, sort, roles[f], things, numbers)
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


def worth(score, roles):
    """what is explained, less what the reading costs: a part for every kind of statement, more for a value from nowhere"""
    return score[0] - 0.25 * len(roles) - 0.75 * sum(r[0] == "to" for r in roles.values())


def learn(task, q, limit=5000):
    """the best way of recalling for a kind of question: every way of looking and every thing to ask of it, the parts of
    the kinds of statements found one at a time (and two at a time where one changes nothing) -> Reading"""
    form = next(f for f, n in task.asked.items() if n == q)
    golds = [g for story in task.stories for f, quest, _, _, g in story if quest and f == q]
    nt = sum(k != "#" for k in form[2])
    nn = len(form[2]) - nt
    shape = "b" if all(g in ("yes", "no") for g in golds) else "n" if all(g.isdigit() for g in golds) else "t"
    owns = Counter(kind(x) for g in golds for x in g.split(",") if thing(x))
    own = owns.most_common(1)[0][0] if owns else None
    consts = sorted({x for g in golds for x in g.split(",") if thing(x)})[:6] if shape != "n" else []
    sizes = {n: (sum(k != "#" for k in f[2]), sum(k == "#" for k in f[2])) for f, n in task.forms.items()}
    builds = {}                                                  # kinds of statements of one build: tried all in one part first
    for f, n in task.forms.items():
        builds.setdefault(f[2], []).append(n)
    best = None
    for sort in ("value", "link", "amount"):
        mine = {f: options(sort, a, b, consts if sort == "value" else ()) for f, (a, b) in sizes.items()}
        mine = {f: v for f, v in mine.items() if v}
        units = [tuple(f for f in fs if f in mine) for fs in builds.values()]
        units = [u for u in units if u]
        for ask in asks(sort, nt, nn, shape, own):
            roles, score, used = {}, grade(run(task, q, sort, ask, {}, own)), 0

            def better(trial):
                nonlocal roles, score, used
                if trial == roles or used > limit:
                    return False
                got = grade(run(task, q, sort, ask, trial, own))
                used += 1
                if worth(got, trial) > worth(score, roles):
                    roles, score = trial, got
                    return True
                return False

            def put(unit, role):
                return {**{k: v for k, v in roles.items() if k not in unit}, **({f: role for f in unit} if role else {})}
            for level in (units, [(f,) for f in mine]):          # first the builds, then every kind of statement by itself
                for sweep in range(4):
                    moved = False
                    for u in level:
                        for role in mine[u[0]] + [None]:
                            moved |= better(put(u, role))
                    if not moved and sweep == 0 and score[0] < len(golds) and 1 < len(level) <= 8:      # nothing helps alone: two at once
                        for u, v in itertools.combinations(level, 2):
                            for a in mine[u[0]][:12]:
                                for b in mine[v[0]][:12]:
                                    base = roles
                                    trial = {**{k: x for k, x in base.items() if k not in u + v}, **{f: a for f in u}, **{f: b for f in v}}
                                    moved |= better(trial)
                    if not moved or score[0] == len(golds):
                        break
                if score[0] == len(golds):
                    break
            if roles and (best is None or worth(score, roles) > worth((best.right,), best.roles)):
                best = Reading(sort, ask, roles, score[1], score[2], score[0], len(golds))
            if best and best.right == len(golds):
                return best
    return best


class Learner:
    def __init__(self, episodes):
        self.task = Task(episodes)
        self.readings = {q: learn(self.task, q) for q in self.task.asked.values()}
        self.usual = {q: Counter(g for story in self.task.stories for f, quest, _, _, g in story if quest and f == q).most_common(1)[0][0] for q in self.task.asked.values()}

    def explained(self):
        return sum(r.right for r in self.readings.values() if r), sum(len([1 for story in self.task.stories for f, quest, _, _, _ in story if quest and f == q]) for q in self.readings)

    def test(self, episodes):
        """-> right, all; and of the kinds of questions it has explained in full: right, all"""
        right = total = sure_right = sure_total = 0
        task = self.task
        for ep in episodes:
            story = []
            for op, quest, things, numbers, kinds, gold in ep:
                f = task.form(op, quest, things, kinds)
                story.append(((task.asked if quest else task.forms).get(f, -1), quest, things, numbers, gold))
            for q, r in self.readings.items():
                for said_, gold in run(task, q, r.sort, r.ask, r.roles, None, [story]) if r else ():
                    told = ("yes" if said_ != r.turned else "no") if said_.__class__ is bool else said_
                    told = r.default if told is None else told
                    right += told == gold
                    total += 1
                    if r.right == r.total:
                        sure_right += told == gold
                        sure_total += 1
                if not r:
                    hit = [g == self.usual[q] for f2, quest, _, _, g in story if quest and f2 == q]
                    right, total = right + sum(hit), total + len(hit)
            unknown = [g for f2, quest, _, _, g in story if quest and f2 == -1]
            total += len(unknown)
        return right, total, sure_right, sure_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--stories", type=int, default=20)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    files = glob.glob(os.path.join(a.data, "qa*_train.txt"))
    tasks = [int(x) for x in a.tasks.split(",") if x] or sorted(int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1)) for f in files)
    for k in tasks:
        t0 = time.time()
        raw = rows.stories(glob.glob(os.path.join(a.data, f"qa{k}_*train.txt"))[0], a.stories)
        best = None
        for reader in rows.guesses(raw):                         # the stories as they are, then with one guess about their words
            m = Learner(reader.rows(raw))                        # (which of them are things): taken if more is explained so
            right, total = m.explained()
            if best is None or right > best[0]:
                best = (right, total, m, reader)
            if right == total:
                break
        right, total, m, reader = best
        full = sum(bool(r) and r.right == r.total for r in m.readings.values())
        print(f"task {k:2d}: learnt from {a.stories} stories, {right}/{total} of their questions right, {full} of {len(m.readings)} kinds of questions explained in full ({time.time() - t0:.1f} s)"
              + (" - " + "; ".join(reader.guessed) if reader.guessed else ""), flush=True)
        if a.show:
            back = {n: f for f, n in m.task.asked.items()}
            for q, r in sorted(m.readings.items()):
                print(f"  ? {back[q][0]}{' [' + back[q][3] + ']' if back[q][3] else ''}: " + (f"{r.right}/{r.total} {r.tell(m.task)}" if r else "-")[:300], flush=True)
        tests = reader.rows(rows.stories(glob.glob(os.path.join(a.data, f"qa{k}_*test.txt"))[0]))
        r, t, sr, st = m.test(tests)
        print(f"test {k:2d}: {r}/{t} = {100 * r / max(1, t):.1f}%; sure of {full} of {len(m.readings)} kinds of questions, {sr}/{st} right on them", flush=True)


if __name__ == "__main__":
    main()
