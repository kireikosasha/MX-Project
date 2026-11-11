#!/usr/bin/env python3
"""
The free-language experiment in one run: for every task the learner learns the template language from 20 stories, the
reader learns to read free wording, and the table of the README is printed.

    python run.py --babi tasks_1-20_v1-2 --new new --out free_out            # reader on all the tasks
    python run.py ... --school                                               # the reader first learns the other tasks
    python run.py ... --plain plain                                          # also writes the files for the transformer

--new is the folder with the sets A, B, C made by newtasks.py. A task already done (its json is in --out) is skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GLOVE = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"

# name in free/stories, where the task is from, its number, when the reader's code first met it
TASKS = [
    ("b1", "en", 1, "dev"), ("b6", "en", 6, "dev"), ("c7", "C", 7, "dev"), ("c9", "C", 9, "dev"), ("c10", "C", 10, "dev"),
    ("b10", "en", 10, "held-out 1"), ("b15", "en", 15, "held-out 1"), ("b17", "en", 17, "held-out 1"),
    ("a1", "A", 1, "held-out 1"), ("a3", "A", 3, "held-out 1"), ("k4", "B", 4, "held-out 1"),
    ("b9", "en", 9, "held-out 2"), ("b14", "en", 14, "held-out 2"), ("b18", "en", 18, "held-out 2"),
    ("b19", "en", 19, "held-out 2"), ("a10", "A", 10, "held-out 2"), ("a11", "A", 11, "held-out 2"),
]


def folder(a, where):
    return os.path.join(a.babi, "en") if where == "en" else os.path.join(a.new, where)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--babi", required=True, help="the folder tasks_1-20_v1-2")
    ap.add_argument("--new", required=True, help="the folder with the new task sets A, B, C")
    ap.add_argument("--out", default="free_out")
    ap.add_argument("--only", default="", help="names of tasks, or a group (dev, held-out 1, held-out 2)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--vectors", default="", help="a GloVe file for the reader's dictionary ('get' fetches the 50-number one)")
    ap.add_argument("--school", action="store_true", help="every task is read after the lessons of all the other tasks")
    ap.add_argument("--plain", default="", help="also write the stories as plain task files for colab/baseline.py here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.vectors == "get":
        a.vectors = os.path.join(a.out, "glove50.gz")
        if not os.path.exists(a.vectors):
            urllib.request.urlretrieve(GLOVE, a.vectors)
    todo = [t for t in TASKS if not a.only or t[0] in a.only.split(",") or t[3] == a.only]
    if a.plain:
        spec = ",".join(f"{n}:{folder(a, w)}:{k}" for n, w, k, _ in todo)
        subprocess.run([sys.executable, os.path.join(HERE, "plain.py"), "--stories", os.path.join(HERE, "stories"), "--tasks", spec,
                        "--out", a.plain], check=True)
    tag = "school_" if a.school else ""
    lessons = os.path.join(a.out, "lessons")
    os.makedirs(lessons, exist_ok=True)
    running = []

    def wait(limit):
        while len(running) > limit:
            running.pop(0).wait()
    for name, where, k, _ in todo:
        out = os.path.join(a.out, f"{tag}{name}.json")
        if os.path.exists(out):
            continue
        cmd = [sys.executable, os.path.join(HERE, "..", "reader.py"), "--data", folder(a, where), "--k", str(k), "--free",
               os.path.join(HERE, "stories"), "--name", name, "--out", out]
        cmd += ["--vectors", a.vectors] if a.vectors else []
        cmd += ["--school", os.path.join(lessons, "*.json")] if a.school else ["--lessons", os.path.join(lessons, name + ".json")]
        wait(a.jobs - 1)
        running.append(subprocess.Popen(cmd, stdout=open(os.path.join(a.out, f"{tag}{name}.log"), "w"), stderr=subprocess.STDOUT))
    wait(0)
    print("\n| task | group | questions | learner alone | reader, no free stories | reader, 40 free stories | best the candidates allow |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    groups = {}
    for name, where, k, group in todo:
        f = os.path.join(a.out, f"{tag}{name}.json")
        if not os.path.exists(f):
            print(f"| {name} | {group} | failed, see {tag}{name}.log |")
            continue
        r = json.load(open(f))
        n = r["test_questions"]
        row = [100 * r["learner_on_free_test"] / n, 100 * r["reader_template_only"][1] / n, 100 * r["reader_with_free"][1] / n,
               100 * r["ceiling_with_answers"] / n]
        groups.setdefault(group, []).append(row)
        print(f"| {name} ({where} {k}) | {group} | {n} | " + " | ".join(f"{x:.1f}" for x in row) + " |")
    for group, rows in groups.items():
        print(f"| mean | {group} | | " + " | ".join(f"{sum(r[i] for r in rows) / len(rows):.1f}" for i in range(4)) + " |")


if __name__ == "__main__":
    main()
