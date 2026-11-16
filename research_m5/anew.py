#!/usr/bin/env python3
"""
StepGame said anew: the same stories, every sentence in the words of other authors.

The sentence patterns in newtasks/phrasings/author_NN.json were written by twelve authors, each in a style of its
own (plain speech, a clock face, compass points, a game board...), for the eight relations of StepGame. A story of
StepGame is taken, of every sentence it is found what it says (by the table that walk.py learns from the answers of
the original stories), and the sentence is written anew with a pattern drawn at random from the patterns of some
authors. The stories to teach on are said in the patterns of one group of authors, the stories of the test in the
patterns of another: not one kind of sentence of the test is met in teaching.

    python anew.py --data .../StepGame/Dataset/TrainVersion --out DIR --teach 1,2,3,4,5,6,7,8 --test 9,10,11,12
"""
import argparse
import glob
import json
import os
import random
import re
from collections import Counter

import torch

import walk

RELATIONS = ("left", "right", "above", "below", "upper-left", "upper-right", "lower-left", "lower-right")


def patterns(folder, authors, dropped=()):
    """the patterns of these authors -> {relation: [pattern]}; a pattern must name both AA and BB, in plain words"""
    out, bad = {r: [] for r in RELATIONS}, 0
    for n in authors:
        data = json.load(open(os.path.join(folder, f"author_{n:02d}.json")))
        for relation in RELATIONS:
            for p in data.get(relation, []):
                text = p.strip() if p.__class__ is str else ""
                good = re.search(r"\bAA\b", text) and re.search(r"\bBB\b", text) and re.fullmatch(r"[A-Za-z0-9 ,.':;\-]+", text)
                if good and (n, relation, text) not in dropped:
                    out[relation].append(text)
                else:
                    bad += 1
    return out, bad


def said(pattern, first, second):
    """a pattern with its two things put in: all the other words made small"""
    text = re.sub(r"\bAA\b", "\x00", pattern)
    text = re.sub(r"\bBB\b", "\x01", text).lower().replace("-", " ").replace(";", ",").replace(":", " ")
    return text.replace("\x00", first).replace("\x01", second)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the folder TrainVersion of StepGame")
    ap.add_argument("--phrasings", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "newtasks", "phrasings"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--teach", default="1,2,3,4,5,6,7,8", help="the authors in whose words the stories to teach on are said")
    ap.add_argument("--test", default="9,10,11,12", help="the authors in whose words the stories of the test are said")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng = random.Random(a.seed)
    dropped = set()
    for f in glob.glob(os.path.join(a.phrasings, "checked_*.json")):          # (patterns another reader took for another relation)
        dropped |= {(n, r, p) for n, r, p in json.load(open(f))}
    pools = {}
    for part, authors in (("teach", a.teach), ("test", a.test)):
        pools[part], bad = patterns(a.phrasings, [int(x) for x in authors.split(",")], dropped)
        print(f"{part}: authors {authors}: {sum(len(v) for v in pools[part].values())} patterns ({bad} left out)", flush=True)

    kinds, asks, answers = {}, {}, {}                            # what every sentence of the original stories says: the table of walk.py
    taught = [x for x in walk.read(os.path.join(a.data, "train.json"), kinds, asks, answers, True) if x[0] is not None]
    model = walk.register(taught, len(kinds), len(asks), len(answers), argparse.Namespace(seed=a.seed, begin=4, steps=300, numbers=2), rng, lambda line: None)
    alone = {}
    for s, q, g in taught:
        if len(s) == 1:
            alone.setdefault(s[0], Counter())[g] += 1
    double = {k for (k, _), c in alone.items() if max(c.values()) < 0.9 * sum(c.values())}
    word = {i: w for w, i in answers.items()}
    print(f"the table of the original sentences explains {100 * sum(model.hard(s, q) == g for s, q, g in taught) / len(taught):.2f}% of the stories; "
          f"{len(double)} kinds used for two things are left out with their stories", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for name, part in [("train.json", "teach")] + [(f"qa{k}_test.json", "test") for k in range(1, 11)]:
        data, out = json.load(open(os.path.join(a.data, name))), {}
        for key, story in data.items():
            sentences = []
            for sentence in story["story"]:
                kind, names = walk.masked(sentence)
                if len(names) != 2 or kind not in kinds or kinds[kind] in double:
                    break
                relation = word[model.hard([(kinds[kind], -1)], 0)]          # how the thing named first stands against the second
                if relation not in pools[part]:
                    break
                sentences.append(said(rng.choice(pools[part][relation]), names[0], names[1]))
            else:
                out[key] = {"story": sentences, "question": story["question"], "label": story["label"]}
        json.dump(out, open(os.path.join(a.out, name), "w"))
        print(f"{name}: {len(out)} stories of {len(data)} said anew", flush=True)


if __name__ == "__main__":
    main()
