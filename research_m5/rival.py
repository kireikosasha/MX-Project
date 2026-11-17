#!/usr/bin/env python3
"""
A rival for walk.py on StepGame: a transformer that reads the words of a story and of its question and names the
answer, taught on the same stories by the same protocol (the 50 000 clean stories of one to five sentences; tried
on the stories of the test with one to ten). It is meant for a video card: on a processor it is a check that the
code runs.

Two forms of it:

  plain    so many layers, each with its own weights
  looped   one block of layers gone through again and again (the input given anew at every turn, the state normed
           after it; the number of turns drawn at random in teaching and set at will afterwards) - the recipe of
           the recurrent-depth models

    python rival.py --data .../StepGame/Dataset/TrainVersion --minutes 60
    python rival.py --data ... --minutes 60 --looped --turns 8
"""
import argparse
import json
import math
import os
import random
import re
import time

import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def words(text):
    return re.findall(r"[A-Z](?='s)|[A-Za-z0-9:]+|'s|[.,?]", text)


def read(path, vocab, answers, grow, most=None):
    """a file of StepGame -> [(the numbers of the words of the story and the question, the number of the answer)].
    The names of the things (single capital letters) are kept as they are; every other word is made small"""
    data, out = json.load(open(path)), []
    for key in list(data)[:most]:
        story = data[key]
        text = ["<q>"] + words(story["question"])
        for sentence in story["story"]:
            text += ["<s>"] + words(sentence)
        text = [w if re.fullmatch(r"[A-Z]|<.>", w) else w.lower() for w in text]
        if grow:
            for w in text:
                vocab.setdefault(w, len(vocab))
            answers.setdefault(story["label"], len(answers))
        out.append(([vocab.get(w, 1) for w in text], answers.get(story["label"], 0)))
    return out


class Rival(nn.Module):
    def __init__(self, vocab, answers, width, layers, heads, looped):
        super().__init__()
        self.embed, self.width, self.looped = nn.Embedding(vocab, width, padding_idx=0), width, looped
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, batch_first=True, norm_first=True)
        self.block = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.adapter = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)
        self.say = nn.Linear(width, answers)

    def place(self, n, shift):
        """the places of the words as waves (so that a story longer than any taught on still has places)"""
        pos = torch.arange(n, device=DEV)[:, None] + shift
        freq = torch.exp(torch.arange(0, self.width, 2, device=DEV) * (-math.log(10000.0) / self.width))
        return torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], -1)

    def forward(self, ids, turns=1):
        pad = ids == 0
        shift = int(torch.randint(0, 200, ())) if self.training else 0             # (in teaching the story begins at a place drawn at random)
        e = self.embed(ids) * math.sqrt(self.width) + self.place(ids.shape[1], shift)[None]
        if not self.looped:
            return self.say(self.norm(self.block(e, src_key_padding_mask=pad))[:, 0])
        s = torch.zeros_like(e)
        for _ in range(turns):
            s = self.norm(self.block(self.adapter(torch.cat([s, e], -1)), src_key_padding_mask=pad))
        return self.say(s[:, 0])


def batch(stories):
    most = max(len(x) for x, _ in stories)
    ids = torch.tensor([x + [0] * (most - len(x)) for x, _ in stories], device=DEV)
    return ids, torch.tensor([g for _, g in stories], device=DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the folder TrainVersion of StepGame")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--looped", action="store_true", help="one block gone through again and again")
    ap.add_argument("--turns", type=int, default=8, help="looped: so many turns on the average in teaching; tried with as many, twice and four times as many")
    ap.add_argument("--stories", type=int, default=0, help="taught on so many stories only (0: on all)")
    ap.add_argument("--tried", type=int, default=2000, help="so many stories of the test for every k")
    ap.add_argument("--save", default="", help="a file to keep the state in every few minutes, and to take the teaching up from")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    rng = random.Random(a.seed)
    vocab, answers = {"<pad>": 0, "<new>": 1}, {}
    taught = read(os.path.join(a.data, "train.json"), vocab, answers, True)
    taught = rng.sample(taught, a.stories) if a.stories else taught
    kept = [read(os.path.join(a.data, f"qa{k}_valid.json"), vocab, answers, False, 500) for k in range(1, 6)]
    layers = max(1, a.layers // 3) if a.looped else a.layers
    model = Rival(len(vocab), len(answers), a.width, layers, a.heads, a.looped).to(DEV)
    print(f"{'looped' if a.looped else 'plain'} transformer: {sum(p.numel() for p in model.parameters())} weights ({layers} layers"
          f"{' in the block' if a.looped else ''}, width {a.width}); taught on {len(taught)} stories, {a.minutes:g} minutes, on {DEV}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0, step, seen, saved = time.time(), 0, 0.0, time.time()
    if a.save and os.path.exists(a.save):                        # (taken up where it was left: the machine may be restarted)
        state = torch.load(a.save, map_location=DEV)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, seen, t0 = state["step"], state["seen"], time.time() - state["spent"]
        print(f"taken up at step {step}, {state['spent'] / 60:.1f} min", flush=True)
    while time.time() - t0 < 60 * a.minutes:
        spent = (time.time() - t0) / (60 * a.minutes)
        for group in opt.param_groups:
            group["lr"] = 3e-4 * min(1.0, step / 500) * (0.1 + 0.9 * math.cos(0.5 * math.pi * spent))
        ids, gold = batch(rng.sample(taught, min(128, len(taught))))
        model.train()
        out = model(ids, max(1, a.turns + rng.randint(-a.turns // 2, a.turns // 2)))
        loss = nn.functional.cross_entropy(out, gold)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        seen = 0.98 * seen + 0.02 * float((out.argmax(-1) == gold).float().mean())
        if step % 500 == 0:
            print(f"step {step}, {(time.time() - t0) / 60:.1f} min: loss {loss.item():.3f}, right on the stories taught on {100 * seen:.0f}%", flush=True)
        if a.save and time.time() - saved > 240:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "seen": seen, "spent": time.time() - t0}, a.save + ".new")
            os.replace(a.save + ".new", a.save)
            saved = time.time()
    if a.save:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "seen": seen, "spent": time.time() - t0}, a.save)
    model.eval()

    def right(stories, turns):
        good = 0
        with torch.no_grad():
            for i in range(0, len(stories), 100):
                ids, gold = batch(stories[i:i + 100])
                good += int((model(ids, turns).argmax(-1) == gold).sum())
        return 100 * good / len(stories)

    print(f"\n{step} steps. Stories kept aside (clean, as those taught on), k = 1..5: " + " ".join(f"{right(s, a.turns):.1f}" for s in kept))
    print("tried on the stories of the test (with sentences that do not matter), right, %, by k:")
    print("%-34s" % "k" + "".join(f"{k:7d}" for k in range(1, 11)))
    tests = [read(os.path.join(a.data, f"qa{k}_test.json"), vocab, answers, False, a.tried) for k in range(1, 11)]
    for turns in ((a.turns, 2 * a.turns, 4 * a.turns) if a.looped else (1,)):
        label = f"looped transformer, {turns} turns" if a.looped else "plain transformer"
        print("%-34s" % label + "".join(f"{right(t, turns):7.1f}" for t in tests), flush=True)


if __name__ == "__main__":
    main()
