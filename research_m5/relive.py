#!/usr/bin/env python3
"""
Dreams from memory: the tasks already learnt, lived again in stories that never were.

A task learnt in full is kept as what it was understood to be: the reading of every kind of sentence and the program
of every kind of question. From that the ground can play any number of new stories - other things do other things,
the questions are answered by the ground itself - and stories of tasks that never were: the same sentences read
another way (the two things of a sentence the other way round, another store, another move of the same build), with
the questions that still have answers. A network (guess.py) is taught on such stories to say at once, from the
stories of a new task, how its sentences are to be read; the exact check stays with the learner.

    python relive.py --tasks new/A:results/rows/rerun_A.txt,... --out lived.pkl      # the tasks as they were understood
"""
import argparse
import glob
import os
import pickle
import random
import re

import memory
import records
import rows

records.Learner.words = True


def told(path):
    """a results file of rows.py --show -> {task: (the codes of its things, what the reader took by what it explains)}"""
    out, k = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"task +(\d+):", line)
        if m:
            k = int(m.group(1))
            out[k] = [{}, ""]
        elif k is not None and line.startswith("  taken by what they explain - "):
            out[k][1] = line.split(" - ", 1)[1].strip()
        elif k is not None and line.startswith("  things: "):
            out[k][0] = dict(x.split("=") for x in line[len("  things: "):].split())
    return out


def again(folder, k, code, taken, read, stories=60):
    """a task as it was understood: its stories as rows, the readings of its sentences (from the results file) and the
    programs of its questions (found again by one fit of the ground) -> the learner, or None if the file and the
    stories do not agree"""
    guesses = dict(x.split(": ", 1) for x in taken.split("; ") if ": " in x)
    pairs = {tuple(x.split("_")) for x in guesses.get("two words name one thing", "").split()}
    alias = dict(x.split("=") for x in guesses.get("words that name the same", "").split())
    raw = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], stories)
    raw = rows.joined(raw, pairs) if pairs else raw
    r = rows.Reader(raw[:20], 0, alias or None)
    r.things, r.code, r.ordered = set(code), dict(code), "the order of the words of an answer matters" in taken
    r.pointers = {w for part in taken.split("; ") if part.startswith("pointers: ") for w in part[len("pointers: "):].split()}
    m = records.Learner(r.rows(raw))
    rec = {}
    for f in m.rec:                                              # (a kind of sentence the task was not learnt from stays unread)
        rec[f] = tuple((a, s, tuple(pick), n) for a, s, pick, n in read.get(memory.key(f[0], f[2]), ()))
    if not any(rec.values()):
        return None
    m.rec = rec
    m.ask = m.fit(rec)[1]
    m.usual = {f: records.Counter(g).most_common(1)[0][0] for f, g in m.forms.items() if f[1]}
    m.reader, m.pairs = r, pairs
    return m


def lived(specs, stories=60, log=print):
    """folder:results file, ... -> the tasks, each as what it was understood to be: the kinds of things with their names,
    the readings, the programs, its own stories and their builds. full: learnt in full, and the programs found again
    answer its own stories as before (only such tasks are dreamt of; the others are kept to try the network on)"""
    out = []
    for spec in specs:
        folder, results = spec.split(":")
        said = told(results)
        for name, full, found in memory.parse(results):
            k = int(name.rsplit(":", 1)[1])
            name = f"{os.path.basename(folder.rstrip('/'))}{k}"
            try:
                m = again(folder, k, said[k][0], said[k][1], dict(found), stories)
            except (KeyError, IndexError, ValueError):
                m = None
            if m is None:
                log(f"{name}: the file and the stories do not agree, left out")
                continue
            right, total = m.run(m.episodes)
            numbers = {}
            for ep in m.episodes:
                for op, quest, _, nums, kinds, _ in ep:
                    for i, v in enumerate(nums):
                        numbers.setdefault(((op, quest, kinds), i), []).append(v)
            out = [t for t in out if t["name"] != name or t["full"] and not (full and 100 * right >= 99 * total)]     # (a task told of twice:
            if any(t["name"] == name for t in out):                                                                    # the telling learnt in
                continue                                                                                               # full is kept, or the last)
            out.append({"name": name, "names": m.names, "rec": m.rec, "ask": m.ask, "episodes": m.episodes, "numbers": numbers,
                        "full": full and 100 * right >= 99 * total, "right": right, "total": total})
            log(f"{name}: {len(m.rec)} kinds of sentences, {len(m.ask)} kinds of questions, {right}/{total}" + ("" if out[-1]["full"] else ", not in full"))
    return out


def play(task, rng, stories=20, rec=None, every=False, own=False, but=()):
    """stories of a task with the answers by the ground. own: its own stories, as they were told; otherwise new ones -
    the builds of its stories (which kinds of sentences, in what order) with other things in them. rec: its sentences
    read so instead of as the task was; every: the questions left without an answer are kept too (with None), so that
    two readings of the same stories can be compared; but: kinds of sentences left out of the stories -> episodes"""
    rec, ask = rec or task["rec"], task["ask"]
    names = {k: rng.sample(v, max(2, min(len(v), rng.randint(max(2, len(v) - 3), len(v))))) for k, v in task["names"].items()}
    out = []
    for told in rng.sample(task["episodes"], min(stories, len(task["episodes"]))):
        s, ep, met = records.State(), [], []
        for op, quest, things, nums, kinds, _ in told:
            form = (op, quest, kinds)
            if form in but:
                continue
            if not own:
                things = []
                for k in kinds:
                    if k != "#":
                        old = [x for x in met if x[0] == k and x not in things]
                        pool = old if old and rng.random() < 0.7 else [x for x in names.get(k, ()) if x not in things]
                        if not pool:
                            break
                        things.append(rng.choice(pool))
                if len(things) < sum(k != "#" for k in kinds):
                    continue
                things = tuple(things)
                nums = tuple(rng.choice(task["numbers"][(form, i)]) for i in range(len(nums)))
            t = things + (op, records.ME)
            s.now = len(ep)
            if quest:
                a = None
                if form in ask:
                    p, flip, default = ask[form]
                    try:
                        a = records.answer(s, p[:-1] + (dict(p[-1]),) if p[0] in records.SPACE else p, t, nums, {})
                    except (KeyError, IndexError, ValueError, TypeError):
                        a = None
                    if a.__class__ is bool:
                        a = "yes" if a != flip else "no"
                    a = default if a is None else a
                    a = a if a.__class__ is str else None
                if a is not None or every:
                    ep.append((op, True, things, nums, kinds, a))
            else:
                for x in things:
                    s.born.setdefault(x, s.now)
                    met.append(x)
                for name, k, pick, n in rec.get(form, ()):
                    try:
                        records.MOVES[name][0](s, k, records.taken(t, nums, pick), (nums[n],) if n is not None else ())
                    except (KeyError, IndexError, ValueError, TypeError):
                        pass
                ep.append((op, False, things, nums, kinds, None))
        out.append(ep)
    return out


def usual(tasks):
    """how often a reading (whatever its stores) was met in the tasks learnt, for every build of a sentence"""
    seen = {}
    for t in tasks:
        for f, moves in t["rec"].items():
            if moves:
                c = seen.setdefault(records.size(f), {})
                key = tuple((name, pick, n) for name, k, pick, n in moves)
                c[key] = c.get(key, 0) + 1
    return seen


BARE = records.Learner([])


def vary(task, rng, seen, rec=None, tries=6):
    """a task that never was: one kind of sentence of a task read another way - a reading its build allows, the ones
    met in the tasks learnt more often than the others. Kept if the stories tell it: the answers are not those of the
    task as it was (rec: or as it was varied before), nor those of the sentence left unread -> the readings, or None"""
    base = rec or task["rec"]
    forms = [f for f, moves in base.items() if moves]
    top = {s: max([k for moves in base.values() for name, k, _, _ in moves if records.MOVES[name][1] == s] + [0]) for s in records.K}
    for _ in range(tries):
        f = rng.choice(forms)
        cands = [p for p in BARE.rows(*records.size(f), top, "FQEL") if p != base[f]]
        if not cands:
            continue
        met = seen.get(records.size(f), {})
        new = rng.choices(cands, [0.2 + met.get(tuple((name, pick, n) for name, k, pick, n in p), 0) for p in cands])[0]
        rec = {**base, f: new}
        seed, own = rng.random(), rng.random() < 0.5
        was, now, bare = (play(task, random.Random(seed), 12, r, True, own) for r in (base, rec, {**base, f: ()}))
        pairs = [(a[5], b[5], c[5]) for x, y, z in zip(was, now, bare) for a, b, c in zip(x, y, z) if a[1]]
        said = [p for p in pairs if p[1] is not None]
        if 10 * len(said) >= 6 * sum(p[0] is not None for p in pairs) and len({p[1] for p in said}) > 1 \
                and 10 * sum(p[1] != p[0] for p in said) >= len(said) and 10 * sum(p[1] != p[2] for p in said) >= len(said):
            return rec
    return None


def dreams(tasks, rng, stories=20, varied=0.5, own=0.5):
    """an endless stream of tasks: a task learnt, or (varied: how often) a task that never was - one or two of its kinds
    of sentences read another way - in its own stories (own: how often) or in new ones, now and then with a kind of
    sentence or of question left out -> (the episodes, the readings of its kinds of sentences, the task it comes from)"""
    seen = usual(tasks)
    while True:
        task, rec = rng.choice(tasks), None
        if rng.random() < varied:
            for _ in range(5):                                   # (a task that can be varied is looked for, a few times)
                rec = vary(task, rng, seen)
                if rec:
                    break
                task = rng.choice(tasks)
            if rec and rng.random() < 0.3:
                rec = vary(task, rng, seen, rec) or rec
        said, asked = [f for f in task["rec"] if task["rec"][f]], list(task["ask"])
        but = ([rng.choice(asked)] if len(asked) > 1 and rng.random() < 0.3 else []) + ([rng.choice(said)] if len(said) > 2 and rng.random() < 0.2 else [])
        n = stories if stories.__class__ is int else rng.randint(*stories)       # (so many stories, or from so many to so many)
        eps = play(task, rng, n, rec, own=rng.random() < own, but=but)
        golds = [r[5] for ep in eps for r in ep if r[1]]
        if len(golds) >= n and len(set(golds)) > 1:
            yield eps, rec or task["rec"], task["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="folder:results file of rows.py --show, comma-separated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    got = lived([x for x in a.tasks.split(",") if x])
    pickle.dump(got, open(a.out, "wb"))
    print(f"{len(got)} tasks as they were understood, {sum(t['full'] for t in got)} of them learnt in full -> {a.out}")


if __name__ == "__main__":
    main()
