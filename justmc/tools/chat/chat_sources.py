#!/usr/bin/env python3
"""
Dialogues for the Millennium chat model from the downloaded datasets (download.py):
  dd_train.zip        DailyDialog (roskoN/dailydialog)                 en, whole dialogues
  persona.parquet     PersonaChat (AlekseyKorshuk/persona-chat)        en, whole dialogues
  everyday.parquet    everyday-conversations (HuggingFaceTB)           en, user <-> assistant
  rudialogues.jsonl   russian_dialogues (Den4ikAI), relevance = 1      ru, question -> answer
  aya.parquet         Aya dataset (CohereLabs), Ukrainian part         uk, question -> answer
An experiment for a server where rough talk is allowed: nothing is filtered but paedophilia.

    python chat_sources.py --raw raw --out src
writes src/dialogs.jsonl  {"lang", "src", "turns": [...]}   (speakers alternate)
       src/pairs.jsonl    {"lang", "src", "m", "r"}
"""
import argparse
import json
import os
import re
import zipfile

import pyarrow.parquet as pq

BAD = re.compile(r"(педоф|paedo|pedophil)", re.I)
URL = re.compile(r"https?://|www\.|@\w|#\w")
MAX_WORDS = 30


def ok(text):
    t = text.strip()
    return 0 < len(t.split()) <= MAX_WORDS and not BAD.search(t) and not URL.search(t)


def clean(t):
    t = re.sub(r"\s+", " ", t.replace("’", "'")).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t


def first_sentences(t, max_words=MAX_WORDS):
    """the first sentences of a long reply that fit in max_words"""
    out = []
    for p in re.split(r"(?<=[.!?])\s+", t.strip()):
        if len(" ".join(out + [p]).split()) > max_words:
            break
        out.append(p)
    return " ".join(out)


def dailydialog(raw):
    z = zipfile.ZipFile(os.path.join(raw, "dd_train.zip"))
    for line in z.read("train/dialogues_train.txt").decode("utf-8", "replace").splitlines():
        yield [clean(u) for u in line.split("__eou__") if u.strip()]


def personachat(raw):
    for r in pq.read_table(os.path.join(raw, "persona.parquet")).to_pylist():
        last = r["utterances"][-1]
        yield [clean(t) for t in last["history"] + [last["candidates"][-1]] if t != "__ SILENCE __"]


def everyday(raw):
    for r in pq.read_table(os.path.join(raw, "everyday.parquet")).to_pylist():
        turns = []
        for m in r["messages"]:
            turns.append(clean(m["content"] if m["role"] == "user" else first_sentences(m["content"])))
        yield turns


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
            yield clean(r["inputs"]), clean(first_sentences(r["targets"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = {}
    with open(os.path.join(a.out, "dialogs.jsonl"), "w", encoding="utf-8") as out:
        for name, fn in (("dailydialog", dailydialog), ("personachat", personachat), ("everyday", everyday)):
            seen = set()
            for turns in fn(a.raw):
                # a dialogue is cut where a turn is unusable; pieces of 2+ turns stay
                piece = []
                for t in turns + [""]:
                    if t and ok(t):
                        piece.append(t)
                        continue
                    key = "\n".join(piece).lower()
                    if len(piece) >= 2 and key not in seen:
                        seen.add(key)
                        out.write(json.dumps({"lang": "en", "src": name, "turns": piece}, ensure_ascii=False) + "\n")
                        n[name] = n.get(name, 0) + 1
                    piece = []
    with open(os.path.join(a.out, "pairs.jsonl"), "w", encoding="utf-8") as out:
        for name, lang, fn in (("rudialogues", "ru", rudialogues), ("aya_uk", "uk", aya_uk)):
            seen = set()
            for m, r in fn(a.raw):
                key = (m.lower(), r.lower())
                if ok(m) and ok(r) and key not in seen:
                    seen.add(key)
                    out.write(json.dumps({"lang": lang, "src": name, "m": m, "r": r}, ensure_ascii=False) + "\n")
                    n[name] = n.get(name, 0) + 1
    print("; ".join(f"{k}: {v}" for k, v in n.items()))


if __name__ == "__main__":
    main()
