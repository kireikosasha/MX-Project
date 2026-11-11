#!/usr/bin/env python3
"""
One run for a machine with a graphics card (Google Colab): is a bigger reader better, and does reading plain
text first help?

Three sizes of the same network (about 0.7, 9 and 85 million parameters), each trained on the word problems
twice: from nothing, and after reading plain text (pretrain.py). Everything else is the same: the exact part,
the problems made by code, the verified problems in live language. The result is a table of SVAMP accuracy.

    python run.py --work /content/drive/MyDrive/mx-logic
Started again, it goes on from what is already done. --quick is a small trial of the whole chain.
"""
import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LOGIC = os.path.dirname(HERE)
SVAMP = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/mawps-asdiv-a_svamp/"
STORIES = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
SIZES = {   # name: reader and head sizes, learning rate, minutes of reading, batch of reading
    "S": {"dim": 96, "hid": 128, "d": 128, "layers": 1, "lr": 2e-3, "minutes": 10, "batch": 512},
    "M": {"dim": 256, "hid": 384, "d": 384, "layers": 2, "lr": 1e-3, "minutes": 40, "batch": 512},
    "L": {"dim": 512, "hid": 1024, "d": 1024, "layers": 3, "lr": 5e-4, "minutes": 60, "batch": 256},
}
QUICK = {
    "S": {"dim": 32, "hid": 32, "d": 32, "layers": 1, "lr": 2e-3, "minutes": 0.7, "batch": 32},
    "M": {"dim": 48, "hid": 48, "d": 48, "layers": 2, "lr": 2e-3, "minutes": 0.7, "batch": 32},
}


def fetch(url, path, size=None):
    """download once; size: only the first so many bytes"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    print("downloading", url, flush=True)
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"} if size else {})
    with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
        while True:
            block = r.read(1 << 22)
            if not block:
                break
            f.write(block)
    os.replace(path + ".part", path)


def finished(log):
    return os.path.exists(log) and "the first 200" in open(log, encoding="utf-8", errors="ignore").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/mx-logic", help="logs, readers and the table are kept here")
    ap.add_argument("--data", default="/content/mx-data", help="downloads and the text turned into ids (large, not worth keeping)")
    ap.add_argument("--text-mb", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    sizes = QUICK if a.quick else SIZES
    if a.quick:
        a.text_mb, a.epochs = 6, 2
    os.makedirs(a.work + "/logs", exist_ok=True)
    os.makedirs(a.data, exist_ok=True)
    train, test, stories = a.data + "/train.csv", a.data + "/dev.csv", a.data + "/stories.txt"
    fetch(SVAMP + "train.csv", train)
    fetch(SVAMP + "dev.csv", test)
    fetch(STORIES, stories, a.text_mb * 1_000_000)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    import torch
    room = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else 0
    print(f"graphics card: {torch.cuda.get_device_name(0) if room else 'none'}, {room:.0f} GB", flush=True)
    if not a.quick:                                              # the reading batch by the memory of the card
        sizes["L"]["batch"] = 384 if room >= 30 else 256 if room >= 20 else 128
        sizes["M"]["batch"] = 1024 if room >= 30 else 512
    many = min(4, max(2, (os.cpu_count() or 2) // 2))            # how many trainings run at once
    world = "300" if a.quick else "2000"

    def solve(name, size, reader=""):
        log = f"{a.work}/logs/{name}.log"
        if finished(log):
            return None
        s = sizes[size]
        cmd = [sys.executable, LOGIC + "/solver.py", "--train", train, "--test", test, "--extra", LOGIC + "/data/variations.jsonl",
               "--heads", "expr", "--world", world, "--sets", "2" if a.quick else "4", "--epochs", str(a.epochs), "--eval-every", "2",
               "--dim", str(s["dim"]), "--hid", str(s["hid"]), "--d", str(s["d"]), "--layers", str(s["layers"]), "--lr", str(s["lr"])]
        if reader:
            cmd += ["--pretrained", reader]
        print("start:", name, flush=True)
        return subprocess.Popen(cmd, cwd=LOGIC, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)

    def wait(procs):
        for name, p in procs:
            if p is not None and p.wait() != 0:
                print(f"!! {name} stopped with an error, see {a.work}/logs/{name}.log", flush=True)

    # from nothing: these do not need the reading, so they go while the readers read
    running = [(f"{k} from nothing", solve(f"{k}_scratch", k)) for k in list(sizes)[:many]]
    later = list(sizes)[many:]
    for k, s in sizes.items():                                    # the readers read, one after another
        out = f"{a.work}/reader_{k}.pt"
        if not os.path.exists(out):
            print(f"== the reader {k} reads plain text for {s['minutes']} minutes", flush=True)
            subprocess.run([sys.executable, LOGIC + "/pretrain.py", "--text", stories, "--text-mb", str(a.text_mb), "--problems", train,
                            "--also", LOGIC + "/data/variations.jsonl", "--world", "2000" if a.quick else "50000",
                            "--words", "8000" if a.quick else "30000", "--out", out, "--cache", a.data + "/text",
                            "--minutes", str(s["minutes"]), "--dim", str(s["dim"]), "--hid", str(s["hid"]), "--layers", str(s["layers"]),
                            "--batch", str(s["batch"]), "--lr", str(s["lr"]), "--save-every", "600"], cwd=LOGIC, env=env)
            if not os.path.exists(out):
                print(f"!! the reader {k} was not made", flush=True)
    wait(running)
    todo = [(f"{k} from nothing", f"{k}_scratch", k, "") for k in later]
    todo += [(f"{k} after reading", f"{k}_read", k, f"{a.work}/reader_{k}.pt") for k in sizes if os.path.exists(f"{a.work}/reader_{k}.pt")]
    print("== training on the problems", flush=True)
    for i in range(0, len(todo), many):
        wait([(title, solve(name, k, reader)) for title, name, k, reader in todo[i:i + many]])
    # the table
    rows = ["| size | parameters | from nothing | after reading plain text |", "| --- | --- | --- | --- |"]
    for k in sizes:
        cells, params = [], "?"
        for kind in ("scratch", "read"):
            text = open(f"{a.work}/logs/{k}_{kind}.log", encoding="utf-8", errors="ignore").read() if os.path.exists(f"{a.work}/logs/{k}_{kind}.log") else ""
            m = re.search(r"chosen by the held out part: epoch (\d+), held out ([\d.]+), test ([\d.]+)", text)
            r = re.search(r"the rest \(800\): ([\d.]+)", text)
            last = re.findall(r"expressions ([\d.]+), ledger", text)
            p = re.search(r"parameters: (\d+)", text)
            if p and kind == "scratch":
                params = f"{int(p.group(1)) / 1e6:.1f} M"
            cells.append(f"{m.group(3)} (held out {m.group(2)}, the 800 unseen {r.group(1) if r else '?'}, last epoch {last[-2] if len(last) > 1 else '?'})" if m else "failed")
        rows.append(f"| {k} | {params} | {cells[0]} | {cells[1]} |")
    reads = []
    for k in sizes:
        if not os.path.exists(f"{a.work}/reader_{k}.pt"):
            continue
        ck = torch.load(f"{a.work}/reader_{k}.pt", map_location="cpu", weights_only=False)
        reads.append(f"reader {k}: read {ck['words_seen'] / 1e6:.0f} million words, names {100 * (ck['right'] or 0):.1f} % of hidden words")
    table = "SVAMP, % right\n\n" + "\n".join(rows) + "\n\n" + "\n".join(reads) + "\n"
    open(a.work + "/results.md", "w", encoding="utf-8").write(table)
    print("\n" + "=" * 70 + "\n" + table + "=" * 70, flush=True)
    print("copy everything between the lines above and send it back", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{(time.time() - t0) / 60:.0f} minutes", flush=True)
