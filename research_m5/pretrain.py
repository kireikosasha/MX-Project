#!/usr/bin/env python3
"""
The reader reads plain text before it sees any problem.

Some words of a passage are hidden and the reader says which words they were. The reader is the one of
solver.py (embedding -> layers of the bidirectional LSTM with the Millennium cell); nobody labels anything,
the text is its own answer. The words, the size and the weights of the reader are saved, and
solver.py --pretrained starts from them.

The text: plain stories (one file, any size; only as much as fits the time is read), and the texts of the
problems themselves without their answers - the training problems, the generated ones, the micro-world.

    python pretrain.py --text stories.txt --problems train.csv --also data/variations.jsonl --out reader.pt --minutes 40
The run is limited by time, not by steps. It saves its weights every few minutes and goes on from them when started again.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
import torch

import mathworld
import solver as SV

DEV = SV.DEV
NUM = re.compile(r"\d[\d,.]*|number\d+")


def pieces(path, limit_mb):
    """the text in pieces of some megabytes, cut at line ends"""
    done = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        while done < limit_mb * 1e6:
            part = f.read(8_000_000)
            if not part:
                break
            part += f.readline()
            done += len(part)
            yield part.replace("<|endoftext|>", " ")


def problem_texts(a):
    """the texts of the problems, without answers"""
    out = []
    if a.problems:
        out += [r["Question"] for r in csv.DictReader(open(a.problems, encoding="utf-8"))] * 3
    if a.also:
        out += [json.loads(line)["text"] for line in open(a.also, encoding="utf-8")] * 3
    rng = random.Random(5)
    out += [" ".join(mathworld.problem(rng)["words"]) for _ in range(a.world)]
    random.Random(6).shuffle(out)
    return out


def corpus(a):
    """-> the words (word -> id), the whole text as one row of ids"""
    cache, names = (a.cache or a.out) + ".text.npy", (a.cache or a.out) + ".words.json"
    if os.path.exists(cache) and os.path.exists(names):
        return json.load(open(names, encoding="utf-8")), np.load(cache)
    t0 = time.time()
    domain = [SV.tokenize(t) for t in problem_texts(a)]
    count, seen = Counter(), 0
    for part in pieces(a.text, a.text_mb):                    # the most frequent words of the beginning of the text
        toks = SV.tokenize(part)
        count.update(toks)
        seen += len(toks)
        if seen > 20_000_000:
            break
    words = [w for w, _ in count.most_common() if not NUM.fullmatch(w)][:a.words]
    words = sorted(set(words) | {w for t in domain for w in t if not NUM.fullmatch(w)})
    ids = {w: i + 2 for i, w in enumerate(words)}             # 0 padding, 1 a number
    vocab = {"ids": ids}
    assert SV.n_vocab(vocab) < 65535
    known = dict(ids)

    def row(toks):
        out = np.empty(len(toks), dtype=np.uint16)
        for i, w in enumerate(toks):
            j = known.get(w)
            if j is None:                                     # a number, or a word without its own vector: its bucket
                j = known[w] = 1 if NUM.fullmatch(w) else SV.bucket(vocab, w)
            out[i] = j
        return out

    rows = [row(t) for t in domain]
    for n, part in enumerate(pieces(a.text, a.text_mb)):
        rows.append(row(SV.tokenize(part)))
        if n % 10 == 0:
            print(f"  text: {sum(len(r) for r in rows) / 1e6:.0f} million words read ({time.time() - t0:.0f} s)", flush=True)
    text = np.concatenate(rows)
    np.save(cache, text)
    json.dump(ids, open(names, "w", encoding="utf-8"))
    return ids, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="plain text")
    ap.add_argument("--text-mb", type=float, default=700, help="how much of it to read, megabytes")
    ap.add_argument("--problems", default="", help="the training problems (csv): their texts are read too")
    ap.add_argument("--also", default="", help="more problems (jsonl with text)")
    ap.add_argument("--world", type=int, default=50000, help="texts of the micro-world")
    ap.add_argument("--words", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="", help="where the text turned into ids is kept (shared by readers of different size)")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hid", type=int, default=384)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hide", type=float, default=0.15, help="share of the words that are hidden")
    ap.add_argument("--save-every", type=float, default=300, help="seconds between saves of the state")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("already there:", a.out)
        return
    torch.manual_seed(a.seed)
    rs = np.random.RandomState(a.seed)
    if DEV.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ids, text = corpus(a)
    n_ids = SV.n_vocab({"ids": ids})
    hidden, n_words = n_ids, n_ids + 1                         # one more id: "a word is hidden here"
    print(f"text: {len(text) / 1e6:.1f} million words, {len(ids)} words with their own vector", flush=True)
    model = SV.Solver(n_words, a.dim, a.hid, 8, 0.1, False, False, False, 0, a.layers).to(DEV)
    reader = [p for n, p in model.named_parameters() if n.split(".")[0] in ("emb", "fw", "bw", "more")]
    proj = torch.nn.Linear(2 * a.hid, a.dim).to(DEV)
    bias = torch.nn.Parameter(torch.zeros(n_words, device=DEV))
    print("reader parameters:", sum(p.numel() for p in reader), "on", DEV, flush=True)
    opt = torch.optim.AdamW(reader + list(proj.parameters()) + [bias], lr=a.lr, weight_decay=1e-2)
    part, spent, step, words_seen = a.out + ".part", 0.0, 0, 0
    if os.path.exists(part):                                   # go on from where it stopped
        ck = torch.load(part, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        proj.load_state_dict(ck["proj"])
        bias.data.copy_(ck["bias"])
        spent, step, words_seen = ck["spent"], ck["step"], ck["words_seen"]
        print(f"going on from minute {spent:.1f}", flush=True)
    offs = np.arange(a.len)
    t0 = last_say = last_save = time.time()
    run_loss = run_acc = None
    model.train()
    while True:
        now = spent + (time.time() - t0) / 60
        if now >= a.minutes:
            break
        for g in opt.param_groups:                             # a short warm-up, then down to a tenth
            g["lr"] = a.lr * min(1.0, (step + 1) / 500) * (1 - 0.9 * now / a.minutes)
        x = torch.from_numpy(text[rs.randint(0, len(text) - a.len, a.batch)[:, None] + offs].astype(np.int64)).to(DEV)
        r = torch.rand(x.shape, device=DEV)
        hide = (r < a.hide) & (x != 0)
        r2 = torch.rand(x.shape, device=DEV)
        seen = torch.where(hide & (r2 < 0.8), torch.full_like(x, hidden), x)                      # mostly: "hidden"
        seen = torch.where(hide & (r2 >= 0.8) & (r2 < 0.9), torch.randint_like(x, 2, n_ids), seen)   # sometimes another word
        h, _ = model.read(seen)
        logits = proj(h[hide]) @ model.emb.weight.t() + bias
        loss = torch.nn.functional.cross_entropy(logits, x[hide])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader, 1.0)
        opt.step()
        step += 1
        words_seen += x.numel()
        acc = (logits.argmax(-1) == x[hide]).float().mean().item()
        if not math.isfinite(loss.item()):
            raise SystemExit("the loss is not a number")
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        run_acc = acc if run_acc is None else 0.98 * run_acc + 0.02 * acc
        if time.time() - last_say > 60:
            last_say = time.time()
            print(f"  minute {now:.0f} of {a.minutes:.0f}: step {step}, {words_seen / 1e6:.0f} million words, loss {run_loss:.3f}, "
                  f"hidden words named right {100 * run_acc:.1f} %", flush=True)
        if time.time() - last_save > a.save_every:
            last_save = time.time()
            torch.save({"model": model.state_dict(), "proj": proj.state_dict(), "bias": bias.data,
                        "spent": now, "step": step, "words_seen": words_seen}, part + ".new")
            os.replace(part + ".new", part)
    state = {k: v.cpu() for k, v in model.state_dict().items() if k.split(".")[0] in ("emb", "fw", "bw", "more")}
    torch.save({"state": state, "ids": ids, "cfg": {"dim": a.dim, "hid": a.hid, "layers": a.layers, "n_words": n_words},
                "words_seen": words_seen, "right": run_acc}, a.out)
    if os.path.exists(part):
        os.remove(part)
    print(f"done: {words_seen / 1e6:.0f} million words read, hidden words named right {100 * (run_acc or 0):.1f} %; saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
