#!/usr/bin/env python3
"""
How well sentences of authors never read are read from their words, by the number of authors learnt from.

The lessons are the sentence patterns of newtasks/phrasings (anew.py): a sentence with its two things taken out and
what it says - how the thing named first stands against the second. In walk.py such lessons come from the table
learnt with answers, or from the register closing on stories without answers; both give them exactly, so here
they are taken as the authors filed them (less the ones a second reader read otherwise). Readers of words (the
small networks of walk.py, five of them, their beliefs averaged) are taught on the patterns of so many authors
and tried on the patterns of four authors kept apart.

    python readers.py
"""
import argparse
import glob
import json
import os
import random

import torch

import anew
import walk

INVERSE = {"left": "right", "right": "left", "above": "below", "below": "above", "upper-left": "lower-right", "lower-right": "upper-left",
           "upper-right": "lower-left", "lower-left": "upper-right"}


def lessons(folder, authors, dropped):
    """the patterns of these authors -> [(the words of the sentence with its things taken out, what it says of the
    thing named first against the second)]"""
    out = []
    for n in authors:
        pool, _ = anew.patterns(folder, [n], dropped)
        for relation, texts in pool.items():
            for text in texts:
                kind, names = walk.masked(anew.said(text, "A", "B"))
                if len(names) == 2:
                    out.append((kind.split(), anew.RELATIONS.index(relation if names[0] == "A" else INVERSE[relation])))
    return out


def right(taught, tried, seed, width):
    """readers taught on the lessons -> the share of the sentences tried that they read rightly, %"""
    belief = 0
    for n in range(5):
        torch.manual_seed(seed + n)
        net = walk.Reader({w for t, _ in taught for w in t}, len(anew.RELATIONS), width)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        texts, gold = [t for t, _ in taught], torch.tensor([g for _, g in taught])
        for step in range(400 + len(taught) // 4):               # (512 lessons at a step, drawn at random, when there are more)
            some = torch.randperm(len(taught))[:512].tolist()
            net.train()
            loss = torch.nn.functional.cross_entropy(net([texts[i] for i in some]), gold[some])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            belief = belief + net([t for t, _ in tried]).softmax(-1)
    return 100 * float((belief.argmax(-1) == torch.tensor([g for _, g in tried])).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrasings", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "newtasks", "phrasings"))
    ap.add_argument("--draws", type=int, default=4, help="so many times the four authors kept apart are drawn anew")
    ap.add_argument("--width", type=int, default=32, help="of the readers")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng = random.Random(a.seed)
    dropped = set()
    for f in glob.glob(os.path.join(a.phrasings, "checked_*.json")):
        dropped |= {(n, r, p) for n, r, p in json.load(open(f))}
    authors = sorted(int(os.path.basename(f)[7:9]) for f in glob.glob(os.path.join(a.phrasings, "author_*.json")))
    counts = [n for n in (1, 2, 4, 8, 12, 16, 20) if n <= len(authors) - 4]
    print(f"{len(authors)} authors; readers of width {a.width}; the share of the sentences of four authors kept apart read rightly, %, by the number of authors learnt from")
    print("%-22s" % "authors learnt from" + "".join(f"{n:8d}" for n in counts))
    table = []
    for draw in range(a.draws):
        apart = rng.sample(authors, 4)
        rest = [n for n in authors if n not in apart]
        rng.shuffle(rest)
        tried = lessons(a.phrasings, apart, dropped)
        print("%-22s" % f"apart: {','.join(map(str, sorted(apart)))}", end="", flush=True)
        row = []
        for n in counts:
            row.append(right(lessons(a.phrasings, rest[:n], dropped), tried, a.seed + 10 * draw, a.width))
            print(f"{row[-1]:8.1f}", end="", flush=True)
        print(flush=True)
        table.append(row)
    print("%-22s" % "mean" + "".join(f"{sum(r[i] for r in table) / len(table):8.1f}" for i in range(len(counts))))


if __name__ == "__main__":
    main()
