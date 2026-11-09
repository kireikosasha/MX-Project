#!/usr/bin/env python3
"""
The whole chat model pipeline in one command (every step is skipped when its result is already there):

  1. download.py        datasets -> <work>/raw
  2. chat_sources.py    dialogue pairs -> <work>/pairs
  3. chat_translate.py  DailyDialog / everyday-conversations en -> ru, uk; russian_dialogues ru -> uk (Opus-MT)
  4. chat_data.py       training sequences -> <work>/data, fine-tuning set (more lines about itself) -> <work>/data_ft
  5. chat_train.py      the model -> <work>/out, fine-tuned -> <work>/out_ft
  6. chat_export.py     JustMC files -> justmc/weights/chat (commit and push them, then ~ai install in the world)

    python tools/chat/run_all.py --work D:/millennium-chat --hidden 128 --epochs 8
    python tools/chat/run_all.py --work D:/millennium-chat --hidden 256 --epochs 12 --tag h256   (a bigger one)

With a CUDA GPU PyTorch uses it (--device cpu to force the processor).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
VOCAB = os.path.join(HERE, "..", "reader", "vocab.json")


def run(args):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-u"] + args)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="folder for datasets, translations, checkpoints")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="suffix of the checkpoint folders (several models side by side)")
    ap.add_argument("--ru-mb", type=int, default=200)
    ap.add_argument("--ru-uk", type=int, default=25000)
    ap.add_argument("--out", default=os.path.join(REPO, "weights", "chat"))
    a = ap.parse_args()
    w = os.path.abspath(a.work)
    dev = ["--device", a.device] if a.device else []
    j = os.path.join
    run([j(HERE, "download.py"), "--out", j(w, "raw"), "--ru-mb", str(a.ru_mb)])
    if not os.path.exists(j(w, "pairs", "rudialogues.jsonl")):
        run([j(HERE, "chat_sources.py"), "--raw", j(w, "raw"), "--out", j(w, "pairs")])
    if not os.path.exists(j(w, "pairs", "rudialogues_uk.jsonl")):
        run([j(HERE, "chat_translate.py"), "--pairs", j(w, "pairs"), "--cache", j(w, "mt"), "--ru-uk", str(a.ru_uk)] + dev)
    common = ["--pairs", j(w, "pairs"), "--mt", j(w, "pairs"), "--qa", j(w, "raw"), "--vocab", VOCAB]
    if not os.path.exists(j(w, "data", "chat_data.npz")):
        run([j(HERE, "chat_data.py")] + common + ["--out", j(w, "data")])
    if not os.path.exists(j(w, "data_ft", "chat_data.npz")):
        run([j(HERE, "chat_data.py")] + common + ["--out", j(w, "data_ft"), "--self-reps", "250", "--sample", "0.15"])
    out, out_ft = j(w, "out" + a.tag), j(w, "out_ft" + a.tag)
    size = ["--hidden", str(a.hidden), "--dim", str(a.dim), "--batch", str(a.batch)]
    if not os.path.exists(j(out, "chat.pt")):
        run([j(HERE, "chat_train.py"), "--data", j(w, "data"), "--out", out, "--epochs", str(a.epochs)] + size + dev)
    if not os.path.exists(j(out_ft, "chat.pt")):
        run([j(HERE, "chat_train.py"), "--data", j(w, "data_ft"), "--out", out_ft, "--init", j(out, "chat.pt"),
             "--epochs", "1", "--lr", "1e-3"] + size + dev)
    run([j(HERE, "chat_export.py"), "--model", j(out_ft, "chat.pt"), "--vocab", VOCAB, "--out", a.out])
    print(f"done: {a.out} - commit and push it, then ~ai install in the world")


if __name__ == "__main__":
    main()
