#!/usr/bin/env python3
"""
The official bAbI tasks (Weston et al., 2015) with the translator + engine: every sentence of a story is
translated into a frame and told to the engine, every question is translated and asked.

    python babi.py --data tasks_1-20_v1-2/en --model translator.pt [--split test] [--show 3]
The data: https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz
"""
import argparse
import glob
import os
import re

import translator as TR
from engine import QUERIES, World, well_formed


def stories(path):
    """-> [[(text, answer or None)]]"""
    out = []
    for line in open(path, encoding="utf-8"):
        n, rest = line.strip().split(" ", 1)
        if n == "1":
            out.append([])
        parts = rest.split("\t")
        out[-1].append((parts[0].strip(), parts[1].strip().lower() if len(parts) > 1 else None))
    return out


def run(model, path, show=0):
    right = total = 0
    wrong = []
    memo = {}
    for story in stories(path):
        world = World()
        for text, gold in story:
            if text not in memo:
                memo[text] = model.parse(text)
            fr = memo[text]
            if gold is None:
                if fr and fr[0] not in QUERIES and well_formed(*fr):
                    world.tell(*fr)
                continue
            ans = None
            if fr and fr[0] in QUERIES and well_formed(*fr):
                try:
                    ans = world.ask(*fr)
                except (KeyError, IndexError):
                    ans = None
            ok = ans == gold or (ans is not None and "," in gold and sorted(ans.split(",")) == sorted(gold.split(",")))
            right += ok
            total += 1
            if not ok and len(wrong) < show:
                wrong.append((text, gold, ans, fr))
    return right, total, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="translator.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--show", type=int, default=0, help="wrong answers to print per task")
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    model = TR.load(a.model)
    files = sorted(glob.glob(os.path.join(a.data, f"qa*_{a.split}.txt")), key=lambda f: int(re.search(r"qa(\d+)_", f).group(1)))
    only = {int(x) for x in a.tasks.split(",") if x}
    tot_r = tot_n = 0
    for f in files:
        k = int(re.search(r"qa(\d+)_", f).group(1))
        if only and k not in only:
            continue
        r, n, wrong = run(model, f, a.show)
        tot_r, tot_n = tot_r + r, tot_n + n
        print(f"task {k:2d} {os.path.basename(f)[len(str(k)) + 3:-len(a.split) - 5]:26s} {r:5d}/{n} = {100 * r / n:6.1f}%")
        for text, gold, ans, fr in wrong:
            print(f"      {text!r}: {ans!r}, expected {gold!r}; frame {fr}")
    print(f"all: {tot_r}/{tot_n} = {100 * tot_r / tot_n:.2f}%")


if __name__ == "__main__":
    main()
