#!/usr/bin/env python3
"""
The learner on the official bAbI tasks in three languages: English, Hindi and English with every letter replaced
(a cipher). Nothing about any of the languages is given. No graphics card is needed.

    python mind.py --work /content/mx-mind
Started again, it goes on from where it stopped.
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/mx-mind")
    ap.add_argument("--runs", default="en:20,hn:20,shuffled:20,en:5,en:10,en:50",
                    help="language:stories of every task to learn from - one run for each pair")
    ap.add_argument("--budget", type=int, default=150, help="seconds for one task")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    base = os.path.join(a.work, "tasks_1-20_v1-2")
    if not os.path.isdir(base):
        print("downloading the tasks", flush=True)
        urllib.request.urlretrieve(DATA, os.path.join(a.work, "babi.tar.gz"))
        with tarfile.open(os.path.join(a.work, "babi.tar.gz")) as t:
            t.extractall(a.work, filter="data")
    runs = [(r.split(":")[0], int(r.split(":")[1])) for r in a.runs.split(",")]
    many = max(1, min(4, (os.cpu_count() or 2)))
    for i in range(0, len(runs), many):                       # several runs at once, one per processor
        procs = []
        for lang, n in runs[i:i + many]:
            name = f"{lang}{n}"
            if os.path.exists(f"{a.work}/{name}.json"):
                continue
            print("start:", name, flush=True)
            procs.append(subprocess.Popen([sys.executable, ROOT + "/acquire.py", "--data", f"{base}/{lang}", "--stories", str(n),
                                           "--budget", str(a.budget), "--state", f"{a.work}/{name}.pkl", "--out", f"{a.work}/{name}.json"],
                                          cwd=ROOT, stdout=open(f"{a.work}/{name}.log", "a"), stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
    rows = ["| language | stories a task | right on the test, % | tasks at 100 % | tasks at 95 % and more |", "| --- | --- | --- | --- | --- |"]
    for lang, n in runs:
        f = f"{a.work}/{lang}{n}.json"
        if not os.path.exists(f):
            rows.append(f"| {lang} | {n} | failed, see {lang}{n}.log | | |")
            continue
        test = json.load(open(f, encoding="utf-8"))["test"]
        right, total = sum(r for _, r, _ in test), sum(t for _, _, t in test)
        rows.append(f"| {lang} | {n} | {100 * right / total:.1f} | {sum(r == t for _, r, t in test)} | {sum(r >= 0.95 * t for _, r, t in test)} |")
    table = "\n".join(rows) + "\n"
    open(a.work + "/results.md", "w", encoding="utf-8").write(table)
    print("\n" + "=" * 70 + "\n" + table + "=" * 70, flush=True)


if __name__ == "__main__":
    main()
