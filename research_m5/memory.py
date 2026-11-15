#!/usr/bin/env python3
"""
What the learner keeps from the tasks it has learnt: for a kind of sentence (its words and where its things stand), the
reading that explained a whole task. A new task begins its search from the readings remembered for the kinds of
sentences it shares with the old ones, instead of from nothing; the search and the exact check stay as they are, so a
remembered reading that does not fit the new stories is dropped like any other guess.

    python memory.py --out memory.json results/rows/rerun_*.txt        # collect from the tasks learnt in full
    python rows.py --data ... --memory memory.json                     # begin from what is remembered
"""
import argparse
import json
import re

from records import MOVES

MOVE = re.compile(r"(\w+) ([FQELTN])(\d+) of \(([^)]*)\)")


def key(op, kinds):
    """a kind of sentence, whatever letters its kinds of things were given"""
    names = {}
    return op + " | " + " ".join("#" if k == "#" else str(names.setdefault(k, len(names))) for k in kinds)


def parse(path):
    """a results file of rows.py --show -> [(task, learnt in full?, [(key, moves)])]"""
    out, name, full, rows = [], None, False, []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"task +(\d+):.* (\d+)/(\d+) of their questions right, (\d+) of (\d+) kinds of statements and (\d+) of (\d+) kinds of questions", line)
        if m:
            name, rows = f"{path}:{m.group(1)}", []
            full = m.group(2) == m.group(3) and m.group(6) == m.group(7)
            continue
        m = re.match(r"test +\d+: (\d+)/(\d+)", line)
        if m and name:
            out.append((name, full and 100 * int(m.group(1)) >= 99 * int(m.group(2)), rows))
            name = None
            continue
        if not name or not line.startswith("    ") or line.startswith("     "):
            continue
        found = list(MOVE.finditer(line))
        if not found:
            continue
        toks = line[4:found[0].start()].split()
        for n in range(1, len(toks)):                            # the sentence, then the kinds of its things and numbers
            op, kinds = toks[:-n], toks[-n:]
            if sum(t in ("*", "#") for t in op) == n and all(re.fullmatch(r"[a-z]|#", k) for k in kinds):
                moves = [[a, int(k), [int(x) for x in picks.replace(" ", "").split(",") if x], 0 if MOVES[a][3] and "#" in kinds else None]
                         for a, _, k, picks in (f.groups() for f in found)]
                rows.append((key(" ".join(op), kinds), moves))
                break
    return out


def collect(paths):
    """-> {kind of sentence: [(reading, the task it comes from)]}, from the tasks learnt in full"""
    memory = {}
    for path in paths:
        for name, full, rows in parse(path):
            for k, moves in rows if full else ():
                if all(m != moves for m, _ in memory.setdefault(k, [])):
                    memory[k].append((moves, name))
    return memory


def recall(memory, forms):
    """the readings remembered for these kinds of sentences -> {form: moves}; where several tasks remember a kind
    differently, the task that shares the most kinds of sentences with these is believed"""
    found = {f: memory.get(key(f[0], f[2]), []) for f in forms}
    votes = {}
    for f, known in found.items():
        for _, task in known:
            votes[task] = votes.get(task, 0) + 1
    out = {}
    for f, known in found.items():
        if known:
            moves, _ = max(known, key=lambda x: votes[x[1]])
            out[f] = tuple((a, k, tuple(pick), n) for a, k, pick, n in moves)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    memory = collect(a.files)
    json.dump(memory, open(a.out, "w"), indent=0, sort_keys=True)
    tasks = {task for known in memory.values() for _, task in known}
    print(f"{len(memory)} kinds of sentences from {len(tasks)} tasks learnt in full")


if __name__ == "__main__":
    main()
