#!/usr/bin/env python3
"""
The comparison: an ordinary transformer trained from nothing on the same bAbI tasks and checked on the same official
test as the learner. It gets the story and the question as words and must choose the answer. Three amounts of data:

    20   the first 20 stories of every task - exactly what the learner learns from
    1k   the whole standard set (1000 questions a task)
    10k  the large set (10000 questions a task)

    python baseline.py --data tasks_1-20_v1-2 --lang en --sizes 20,1k,10k --out base_en.json
One network for all twenty tasks. A tenth of the training stories is held out to choose when to stop.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acquire import read  # noqa: E402

MAX_SENTS = 60          # the last sentences of a story the network sees
PAD, CLS, SEP, ASK, UNK = 0, 1, 2, 3, 4


def examples(stories, task):
    """every question with the statements before it: (task, sentences, question, answer)"""
    out = []
    for story in stories:
        told = []
        for words, gold in story:
            if gold is None:
                told.append(words)
            else:
                out.append((task, tuple(told[-MAX_SENTS:]), words, ",".join(gold)))
    return out


def load(root, lang, size):
    """-> training stories of every task (kept whole, to hold some out), and the test questions"""
    train, test = {}, []
    here = os.path.join(root, lang) if lang else root            # no language: the task files lie in the folder itself
    for k in sorted(int(re.match(r"qa(\d+)_", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(here, "qa*_train.txt"))):
        folder = here + "-10k" if size == "10k" else here
        f = glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0]
        train[k] = read(f, int(size) if size.isdigit() else None)
        test += examples(read(glob.glob(os.path.join(here, f"qa{k}_*test.txt"))[0]), k)
    return train, test


class Net(nn.Module):
    def __init__(self, n_words, n_answers, width=192, layers=4, heads=4, max_len=1024, drop=0.1):
        super().__init__()
        self.word = nn.Embedding(n_words, width, padding_idx=PAD)
        self.place = nn.Embedding(max_len, width)                 # where the word stands
        self.age = nn.Embedding(MAX_SENTS + 2, width)             # how many sentences ago it was said
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, drop, activation="gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, n_answers)
        self.drop = nn.Dropout(drop)

    def forward(self, words, ages):
        pos = torch.arange(words.shape[1], device=words.device).clamp(max=self.place.num_embeddings - 1)
        x = self.drop(self.word(words) + self.place(pos)[None] + self.age(ages))
        x = self.body(x, src_key_padding_mask=words == PAD)
        return self.out(self.norm(x[:, 0]))


def encode(ex, vocab):
    _, told, quest, _ = ex
    words, ages = [CLS], [0]
    for i, s in enumerate(told):
        words += [vocab.get(w, UNK) for w in s] + [SEP]
        ages += [min(len(told) - i, MAX_SENTS) + 1] * (len(s) + 1)
    words += [ASK] + [vocab.get(w, UNK) for w in quest]
    ages += [1] * (len(quest) + 1)
    return words, ages


def batches(data, vocab, answers, size, shuffle, device):
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
        order = [i for part in (sorted(order[j:j + size * 50], key=lambda i: len(data[i][1])) for j in range(0, len(order), size * 50)) for i in part]
    else:
        order.sort(key=lambda i: len(data[i][1]))
    packs = [order[j:j + size] for j in range(0, len(order), size)]
    if shuffle:
        random.shuffle(packs)
    for pack in packs:
        enc = [encode(data[i], vocab) for i in pack]
        n = max(len(w) for w, _ in enc)
        words = torch.tensor([w + [PAD] * (n - len(w)) for w, _ in enc], device=device)
        ages = torch.tensor([a + [0] * (n - len(a)) for _, a in enc], device=device)
        gold = torch.tensor([answers.get(data[i][3], -1) for i in pack], device=device)
        yield pack, words, ages, gold


def evaluate(net, data, vocab, answers, device):
    """-> right answers for every task: {task: [right, all]}"""
    net.eval()
    got = {}
    with torch.no_grad():
        for pack, words, ages, gold in batches(data, vocab, answers, 256, False, device):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                best = net(words, ages).argmax(-1)
            for i, b, g in zip(pack, best.tolist(), gold.tolist()):
                r = got.setdefault(data[i][0], [0, 0])
                r[0] += b == g
                r[1] += 1
    return got


def run(root, lang, size, seed, device, width, layers, max_minutes):
    random.seed(seed)
    torch.manual_seed(seed)
    stories, test = load(root, lang, size)
    train, held = [], []
    for k, st in stories.items():
        st = list(st)
        random.Random(seed).shuffle(st)
        cut = max(1, len(st) // 10)
        held += examples(st[:cut], k)
        train += examples(st[cut:], k)
    vocab = {w: i + 5 for i, w in enumerate(sorted({w for _, told, q, _ in train + held for s in told + (q,) for w in s}))}
    answers = {a: i for i, a in enumerate(sorted({a for *_, a in train + held}))}
    net = Net(len(vocab) + 5, len(answers), width, layers).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    batch = 64
    per_epoch = math.ceil(len(train) / batch)
    steps = max(3000, min(60000, per_epoch * 40))               # little data: many passes; much data: at most 40 passes
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    warm = max(100, steps // 20)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    step, best, best_state, t0, since = 0, -1.0, None, time.time(), 0
    check = max(100, min(per_epoch, 1000))
    print(f"{lang} {size}: {len(train)} questions to learn from, {len(held)} held out, {n_par / 1e6:.1f}M weights, {steps} steps", flush=True)
    done = False
    while not done:
        for _, words, ages, gold in batches(train, vocab, answers, batch, True, device):
            net.train()
            lr = 3e-4 * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(net(words, ages), gold)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % check == 0 or step == steps:
                got = evaluate(net, held, vocab, answers, device)
                acc = sum(r for r, _ in got.values()) / max(1, sum(n for _, n in got.values()))
                since += 1
                if acc > best:
                    best, since = acc, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                print(f"  step {step}: loss {loss.item():.3f}, held out {100 * acc:.1f}%  ({time.time() - t0:.0f} s)", flush=True)
                if step >= steps or since >= 12 or time.time() - t0 > 60 * max_minutes:
                    done = True
                    break
    net.load_state_dict(best_state)
    got = evaluate(net, test, vocab, answers, device)
    rows = [[k, got[k][0], got[k][1]] for k in sorted(got)]
    right, total = sum(r for _, r, _ in rows), sum(n for _, _, n in rows)
    print(f"{lang} {size}: test {right}/{total} = {100 * right / total:.2f}%, tasks at 95% and more: {sum(r >= 0.95 * n for _, r, n in rows)}", flush=True)
    return {"lang": lang, "size": size, "questions": len(train) + len(held), "weights": n_par, "steps": step, "held_out": best, "test": rows,
            "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the folder tasks_1-20_v1-2")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--sizes", default="20,1k,10k")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=90, help="the longest one training may take")
    ap.add_argument("--out", default="baseline.json")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else []
    for size in a.sizes.split(","):
        if any(r["lang"] == a.lang and r["size"] == size for r in done):
            continue
        done.append(run(a.data, a.lang, size, a.seed, device, a.width, a.layers, a.minutes))
        json.dump(done, open(a.out, "w", encoding="utf-8"))
    print("\n| data | questions | right on the test, % | tasks at 95 % and more |\n| --- | --- | --- | --- |")
    for r in done:
        right, total = sum(x for _, x, _ in r["test"]), sum(n for _, _, n in r["test"])
        print(f"| transformer, {r['lang']} {r['size']} | {r['questions']} | {100 * right / total:.1f} | {sum(x >= 0.95 * n for _, x, n in r['test'])} |")


if __name__ == "__main__":
    main()
