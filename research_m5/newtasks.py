#!/usr/bin/env python3
"""
New tasks, not from bAbI: every folder newtasks/<set>/tNN holds a generator gen.py (N stories, seed -> text in the
bAbI format). This script runs the generators and writes the files the learner and the transformer read:

    python newtasks.py --set A --out data_new/A
    python acquire.py --data data_new/A --stories 20 --apart --tasks 1,2,3,4,5,6,7,8,9,10,11,12

Set A was used while the ground was being widened; set B was written by other authors to the same themes, was not
looked at, and is run once on the finished code.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BAD = re.compile(r"\b(os|subprocess|socket|shutil|pathlib|urllib|requests|http|ctypes|importlib|pickle)\b|__import__|\bopen\s*\(|\beval\s*\(|\bexec\s*\(")


def check(text):
    """a generator may only compute and print"""
    hit = BAD.search(text)
    imports = set(re.findall(r"^\s*(?:import|from)\s+([\w.]+)", text, re.M))
    extra = imports - {"random", "sys", "itertools", "collections", "string", "math", "typing", "copy", "functools", "dataclasses", "enum"}
    return (hit.group(0) if hit else None) or (", ".join(sorted(extra)) if extra else None)


def stories(text):
    """the lines of a generator -> checked text; a broken line raises"""
    out, n = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        num, rest = line.split(" ", 1)
        n = n + 1 if int(num) != 1 else 1
        assert int(num) == n, f"numbering: {line!r}"
        if "\t" in rest:
            q, a = rest.split("\t")[:2]
            assert q.rstrip().endswith("?") and a.strip() and a == a.lower(), f"question: {line!r}"
        else:
            assert rest.rstrip().endswith("."), f"statement: {line!r}"
        out.append(line.rstrip())
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=2000, help="stories in the training file (the learner takes the first 20)")
    ap.add_argument("--test", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = os.path.join(HERE, "newtasks", a.set)
    for name in sorted(os.listdir(root)):
        gen = os.path.join(root, name, "gen.py")
        if not os.path.exists(gen):
            print(name, "- no generator")
            continue
        why = check(open(gen, encoding="utf-8").read())
        if why:
            print(name, "- not run, it uses:", why)
            continue
        k = int(name[1:])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for part, n, seed in (("train", a.train, 1), ("test", a.test, 2)):
                    r = subprocess.run([sys.executable, "-I", gen, str(n), str(seed)], capture_output=True, text=True, timeout=300, cwd=tmp)
                    assert r.returncode == 0, r.stderr[-300:]
                    open(os.path.join(a.out, f"qa{k}_{name}_{part}.txt"), "w", encoding="utf-8").write(stories(r.stdout))
            print(name, "- written")
        except Exception as e:                                   # noqa: BLE001
            print(name, "- failed:", str(e)[:200])


if __name__ == "__main__":
    main()
