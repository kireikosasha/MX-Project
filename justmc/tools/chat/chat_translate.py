#!/usr/bin/env python3
"""
Ukrainian (and more Russian) dialogues by machine translation with the Opus-MT models (Marian):
DailyDialog and everyday-conversations en -> ru, uk; part of russian_dialogues ru -> uk. Every sentence is
translated once (consecutive pairs of a dialogue share sentences); a cache file makes the job resumable.

    python chat_translate.py --pairs pairs --cache mt --ru-uk 25000
writes pairs/<source>_<lang>.jsonl
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import MarianMTModel, MarianTokenizer

MODELS = {("en", "ru"): "Helsinki-NLP/opus-mt-en-ru", ("en", "uk"): "Helsinki-NLP/opus-mt-en-uk",
          ("ru", "uk"): "Helsinki-NLP/opus-mt-ru-uk"}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def translate(sents, pair, cache_dir, batch=48):
    cache_path = os.path.join(cache_dir, f"cache_{pair[0]}_{pair[1]}.jsonl")
    done = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            r = json.loads(l)
            done[r["s"]] = r["t"]
    todo = sorted({s for s in sents if s not in done}, key=len)
    if todo:
        tok = MarianTokenizer.from_pretrained(MODELS[pair], cache_dir=cache_dir)
        m = MarianMTModel.from_pretrained(MODELS[pair], cache_dir=cache_dir).eval()
        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as out, torch.no_grad():
            for i in range(0, len(todo), batch):
                src = todo[i:i + batch]
                b = tok(src, return_tensors="pt", padding=True, truncation=True, max_length=96)
                gen = m.generate(**b, num_beams=1, max_new_tokens=80)
                for s, t in zip(src, tok.batch_decode(gen, skip_special_tokens=True)):
                    done[s] = t
                    out.write(json.dumps({"s": s, "t": t}, ensure_ascii=False) + "\n")
                if (i // batch) % 50 == 0:
                    print(f"  {pair}: {i + len(src)}/{len(todo)} ({(i + len(src)) / (time.time() - t0):.0f}/s)", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ru-uk", type=int, default=25000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.cache, exist_ok=True)
    jobs = []
    for name in ("everyday", "dailydialog"):
        for tgt in ("ru", "uk"):
            jobs.append((name, load(os.path.join(a.pairs, name + ".jsonl")), ("en", tgt)))
    ru = load(os.path.join(a.pairs, "rudialogues.jsonl"))
    random.Random(4).shuffle(ru)
    jobs.append(("rudialogues", ru[:a.ru_uk], ("ru", "uk")))
    for name, rows, pair in jobs:
        tr = translate([s for r in rows for s in (r["m"], r["r"])], pair, a.cache)
        with open(os.path.join(a.pairs, f"{name}_{pair[1]}.jsonl"), "w", encoding="utf-8") as out:
            for r in rows:
                out.write(json.dumps({"lang": pair[1], "src": f"{name}_{pair[1]}", "m": tr[r["m"]], "r": tr[r["r"]]},
                                     ensure_ascii=False) + "\n")
        print(f"{name} -> {pair[1]}: {len(rows)} pairs", flush=True)


if __name__ == "__main__":
    main()
