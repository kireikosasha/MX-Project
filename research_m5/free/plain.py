#!/usr/bin/env python3
"""
The free stories as plain task files, for the transformer (colab/baseline.py): for every task the 20 template stories
the learner learns from, then the free stories with answers; the test is the free test stories.

    python plain.py --stories stories --tasks b1:tasks_1-20_v1-2/en:1,c7:new/C:7 --out plain
    python ../colab/baseline.py --data plain --lang "" --sizes 60
"""
import argparse
import glob
import json
import os


def lines(story):
    return "".join(f"{i + 1} {s}\n" if a is None else f"{i + 1} {s}\t{a}\t1\n" for i, (s, a) in enumerate(story))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", default="stories")
    ap.add_argument("--tasks", required=True, help="name:folder:number, ...")
    ap.add_argument("--out", default="plain")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n, spec in enumerate(a.tasks.split(",")):
        name, folder, k = spec.split(":")
        template, at = [], 0
        for line in open(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], encoding="utf-8"):
            at += line.split(" ", 1)[0] == "1"
            if at > 20:
                break
            template.append(line)
        free = {"train": {}, "test": {}}
        for f in sorted(glob.glob(os.path.join(a.stories, name + "_*.json"))):
            for st in json.load(open(f, encoding="utf-8"))["stories"]:
                free[st["part"]][st["id"]] = lines([(x.get("p") or x["s"], x["a"]) for x in st["sentences"]])
        with open(os.path.join(a.out, f"qa{n + 1}_{name}_train.txt"), "w", encoding="utf-8") as f:
            f.write("".join(template) + "".join(v for _, v in sorted(free["train"].items())))
        with open(os.path.join(a.out, f"qa{n + 1}_{name}_test.txt"), "w", encoding="utf-8") as f:
            f.write("".join(v for _, v in sorted(free["test"].items())))
        print(name, "template lines", len(template), "free stories", len(free["train"]), len(free["test"]))


if __name__ == "__main__":
    main()
