#!/usr/bin/env python3
"""
Dialogue pairs (message -> reply) for the Millennium chat model, from the downloaded datasets:
  dd_train.zip        DailyDialog (roskoN/dailydialog)                 en, multi-turn
  persona.parquet     PersonaChat (AlekseyKorshuk/persona-chat)        en, multi-turn
  everyday.parquet    everyday-conversations (HuggingFaceTB)           en, user <-> assistant
  rudialogues.jsonl   russian_dialogues (Den4ikAI), relevance = 1      ru, question -> answer
  aya.parquet         Aya dataset (CohereLabs), Ukrainian part         uk
Short pairs only; everything stays except paedophilia.

    python chat_sources.py --raw raw --out pairs
writes pairs/<source>.jsonl  {"lang", "src", "m", "r"}
"""
import argparse
import io
import json
import os
import random
import re
import zipfile

import pyarrow.parquet as pq

# an experiment: rough talk, swearing and anything else stay; only paedophilia is dropped
BAD = re.compile(r"(педоф|paedo|pedophil)", re.I)
URL = re.compile(r"https?://|www\.|@\w|#\w")


def ok(text, max_words):
    t = text.strip()
    return 0 < len(t.split()) <= max_words and not BAD.search(t) and not URL.search(t)


def clean(t):
    t = re.sub(r"\s+", " ", t.replace("’", "'")).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t


def first_sentences(t, max_words):
    """the first sentences of a long assistant reply that fit in max_words"""
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    out = []
    for p in parts:
        if len(" ".join(out + [p]).split()) > max_words:
            break
        out.append(p)
    return " ".join(out)


def dailydialog(raw):
    z = zipfile.ZipFile(os.path.join(raw, "dd_train.zip"))
    text = z.read("train/dialogues_train.txt").decode("utf-8", "replace")
    for line in text.splitlines():
        turns = [clean(u) for u in line.split("__eou__") if u.strip()]
        for a, b in zip(turns, turns[1:]):
            yield a, b


def personachat(raw):
    for r in pq.read_table(os.path.join(raw, "persona.parquet")).to_pylist():
        last = r["utterances"][-1]
        hist = last["history"] + [last["candidates"][-1]]
        for a, b in zip(hist, hist[1:]):
            if a != "__ SILENCE __":
                yield clean(a), clean(b)


def everyday(raw):
    for r in pq.read_table(os.path.join(raw, "everyday.parquet")).to_pylist():
        msgs = r["messages"]
        for a, b in zip(msgs, msgs[1:]):
            if a["role"] == "user" and b["role"] == "assistant":
                yield clean(a["content"]), clean(first_sentences(b["content"], 30))


def rudialogues(raw):
    with open(os.path.join(raw, "rudialogues.jsonl"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("relevance") == 1 and r.get("question") and r.get("answer"):
                yield clean(r["question"]), clean(r["answer"])


def aya_uk(raw):
    t = pq.read_table(os.path.join(raw, "aya.parquet"), columns=["inputs", "targets", "language"])
    for r in t.to_pylist():
        if r["language"] == "Ukrainian":
            yield clean(r["inputs"]), clean(first_sentences(r["targets"], 30))


SOURCES = {"dailydialog": ("en", dailydialog), "personachat": ("en", personachat), "everyday": ("en", everyday),
           "rudialogues": ("ru", rudialogues), "aya_uk": ("uk", aya_uk)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-words", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, (lang, fn) in SOURCES.items():
        seen, n = set(), 0
        with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8") as out:
            for m, r in fn(a.raw):
                if not (ok(m, a.max_words) and ok(r, a.max_words)):
                    continue
                key = (m.lower(), r.lower())
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps({"lang": lang, "src": name, "m": m, "r": r}, ensure_ascii=False) + "\n")
                n += 1
        print(f"{name}: {n} pairs")


if __name__ == "__main__":
    main()
