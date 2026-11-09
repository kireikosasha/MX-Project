#!/usr/bin/env python3
"""
More Russian and Ukrainian dialogues by machine translation with the Opus-MT models (Marian):
DailyDialog and everyday-conversations en -> ru, uk (whole dialogues); part of russian_dialogues ru -> uk.
Every sentence is translated once; a cache file makes the job resumable.

    python chat_translate.py --src src --cache mt [--device cuda]
writes src/dialogs_ru.jsonl, src/dialogs_uk.jsonl, src/pairs_uk.jsonl

The step is optional: if the models cannot be loaded (transformers needs a newer PyTorch, no network ...)
it says so and exits with code 3; the data builder then works with the original languages and the
synthetic dialogues only.
"""
import argparse
import json
import os
import random
import sys
import time

MODELS = {("en", "ru"): "Helsinki-NLP/opus-mt-en-ru", ("en", "uk"): "Helsinki-NLP/opus-mt-en-uk",
          ("ru", "uk"): "Helsinki-NLP/opus-mt-ru-uk"}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def translate(sents, pair, cache_dir, dev, batch):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    cache_path = os.path.join(cache_dir, f"cache_{pair[0]}_{pair[1]}.jsonl")
    done = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue                      # a line cut off by an interrupted run
            done[r["s"]] = r["t"]
    todo = sorted({s for s in sents if s not in done}, key=len)
    if todo:
        tok = MarianTokenizer.from_pretrained(MODELS[pair], cache_dir=cache_dir)
        m = MarianMTModel.from_pretrained(MODELS[pair], cache_dir=cache_dir).eval().to(dev)
        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as out, torch.no_grad():
            for i in range(0, len(todo), batch):
                src = todo[i:i + batch]
                b = tok(src, return_tensors="pt", padding=True, truncation=True, max_length=96).to(dev)
                gen = m.generate(**b, num_beams=1, max_new_tokens=80)
                for s, t in zip(src, tok.batch_decode(gen, skip_special_tokens=True)):
                    done[s] = t
                    out.write(json.dumps({"s": s, "t": t}, ensure_ascii=False) + "\n")
                if (i // batch) % 50 == 0:
                    print(f"  {pair[0]} -> {pair[1]}: {i + len(src)}/{len(todo)} ({(i + len(src)) / (time.time() - t0):.0f}/s)", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ru-uk", type=int, default=25000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    try:
        import torch
        from transformers import MarianMTModel  # noqa: F401
        MarianMTModel.from_pretrained               # raises when transformers has no usable PyTorch
    except Exception as e:  # noqa: BLE001
        print(f"translation skipped: {str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__}")
        sys.exit(3)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    os.makedirs(a.cache, exist_ok=True)
    batch = 128 if dev.startswith("cuda") else 48
    dialogs = [d for d in load(os.path.join(a.src, "dialogs.jsonl")) if d["src"] in ("dailydialog", "everyday")]
    pairs = [p for p in load(os.path.join(a.src, "pairs.jsonl")) if p["src"] == "rudialogues"]
    random.Random(4).shuffle(pairs)
    pairs = pairs[:a.ru_uk]
    try:
        for tgt in ("ru", "uk"):
            tr = translate([t for d in dialogs for t in d["turns"]], ("en", tgt), a.cache, dev, batch)
            with open(os.path.join(a.src, f"dialogs_{tgt}.jsonl.part"), "w", encoding="utf-8") as out:
                for d in dialogs:
                    out.write(json.dumps({"lang": tgt, "src": f"{d['src']}_{tgt}", "turns": [tr[t] for t in d["turns"]]},
                                         ensure_ascii=False) + "\n")
            os.replace(os.path.join(a.src, f"dialogs_{tgt}.jsonl.part"), os.path.join(a.src, f"dialogs_{tgt}.jsonl"))
            print(f"dialogues -> {tgt}: {len(dialogs)}", flush=True)
        tr = translate([s for p in pairs for s in (p["m"], p["r"])], ("ru", "uk"), a.cache, dev, batch)
        with open(os.path.join(a.src, "pairs_uk.jsonl.part"), "w", encoding="utf-8") as out:
            for p in pairs:
                out.write(json.dumps({"lang": "uk", "src": "rudialogues_uk", "m": tr[p["m"]], "r": tr[p["r"]]},
                                     ensure_ascii=False) + "\n")
        os.replace(os.path.join(a.src, "pairs_uk.jsonl.part"), os.path.join(a.src, "pairs_uk.jsonl"))
        print(f"russian_dialogues -> uk: {len(pairs)}", flush=True)
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        print(f"translation skipped: {str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__}")
        sys.exit(3)


if __name__ == "__main__":
    main()
