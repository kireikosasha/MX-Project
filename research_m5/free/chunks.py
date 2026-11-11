#!/usr/bin/env python3
"""
Cuts a task into files of 20 stories to be reworded in free language (see brief.md), and checks the reworded files.

    python chunks.py cut   --data tasks_1-20_v1-2/en --k 15 --name b15 --out src
    python chunks.py check --src src --done stories --name b15

The free stories to learn from are the training stories 21-60 (the learner itself learns from stories 1-20 in template
language); the free test is the first 80 test stories.
"""
import argparse
import glob
import json
import os
import re


def stories(path):
    out = []
    for line in open(path, encoding="utf-8"):
        n, rest = line.strip().split(" ", 1)
        if n == "1":
            out.append([])
        parts = rest.split("\t")
        out[-1].append({"s": parts[0].strip(), "a": parts[1].strip().lower() if len(parts) > 1 else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cut", "check"])
    ap.add_argument("--data")
    ap.add_argument("--k", type=int)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="src")
    ap.add_argument("--src", default="src")
    ap.add_argument("--done", default="stories")
    ap.add_argument("--first", type=int, default=20, help="the first training story to take (those before it are the template ones)")
    ap.add_argument("--per", type=int, default=20, help="stories in a file")
    a = ap.parse_args()
    if a.what == "cut":
        os.makedirs(a.out, exist_ok=True)
        train = stories(glob.glob(os.path.join(a.data, f"qa{a.k}_*train.txt"))[0])[a.first:a.first + 40]
        test = stories(glob.glob(os.path.join(a.data, f"qa{a.k}_*test.txt"))[0])[:80]
        both = [{"part": "train", "id": i, "sentences": s} for i, s in enumerate(train)] + [{"part": "test", "id": i, "sentences": s} for i, s in enumerate(test)]
        for c in range(0, len(both), a.per):
            json.dump({"task": a.name, "stories": both[c:c + a.per]}, open(os.path.join(a.out, f"{a.name}_{c // a.per:02d}.json"), "w", encoding="utf-8"), indent=0)
        print(a.name, len(both), "stories,", sum(len(s["sentences"]) for s in both), "sentences")
        return
    bad = 0
    for f in sorted(glob.glob(os.path.join(a.src, a.name + "_*.json"))):
        g = os.path.join(a.done, os.path.basename(f))
        if not os.path.exists(g):
            print("missing", g)
            bad += 1
            continue
        src, done = json.load(open(f, encoding="utf-8")), json.load(open(g, encoding="utf-8"))
        if len(src["stories"]) != len(done["stories"]):
            print(g, "stories", len(done["stories"]), "of", len(src["stories"]))
            bad += 1
            continue
        for x, y in zip(src["stories"], done["stories"]):
            same = (x["part"], x["id"]) == (y["part"], y["id"]) and len(x["sentences"]) == len(y["sentences"]) and all(
                u["s"] == v["s"] and u["a"] == v["a"] and v.get("p") and (v["p"].strip()[-1] == "?") == (u["a"] is not None)
                and sorted(re.findall(r"\d+", u["s"])) == sorted(re.findall(r"\d+", v["p"])) for u, v in zip(x["sentences"], y["sentences"]))
            if not same:
                print(g, "story", x["part"], x["id"], "does not match")
                bad += 1
    print(a.name, "bad:", bad)


if __name__ == "__main__":
    main()
