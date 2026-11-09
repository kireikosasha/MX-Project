#!/usr/bin/env python3
"""
Dataset of the Millennium reader: WordPiece vocabulary for English, Russian and Ukrainian and
training examples that look exactly like what the JustMC code feeds the model (reader_text.py):
the two sentences of the paragraph that share most keywords with the question, at most MAXC tokens.
An answer outside that window, an unanswerable question (SQuAD 2.0, UA-SQuAD) and a question paired
with an unrelated paragraph are "no answer" examples.

    python build_reader_data.py --raw raw --out data --vocab 20000
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_text as R  # noqa: E402


def _cum(parts):
    acc, res = 0, []
    for p in parts:
        res.append(acc)
        acc += len(p)
    res.append(acc)
    return res


def norm_text(t):
    parts = [R.normalize(ch) for ch in t]
    return "".join(parts), _cum(parts)


def add(items, lang, split, q, ctx, answers):
    """answers: [(text, start)] in the original context, [] = unanswerable"""
    c, cm = norm_text(ctx)
    qn, _ = norm_text(q)
    spans = []
    for text, st in answers:
        if st is None or st < 0:
            st = ctx.find(text)
            if st < 0:
                continue
        a = cm[st]
        b = cm[min(len(ctx), st + len(text))]
        if b > a:
            spans.append((a, b))
    items.append({"lang": lang, "split": split, "q": qn, "ctx": c, "spans": spans, "impossible": not answers})


def load(raw):
    items = []
    for name, lang, split in (("squad_train", "en", "train"), ("squad_val", "en", "val"),
                              ("sber_train", "ru", "train"), ("sber_val", "ru", "val"),
                              ("uahplt_train", "uk", "train"), ("uahplt_test", "uk", "val")):
        for r in pq.read_table(os.path.join(raw, name + ".parquet")).to_pylist():
            a = r["answers"]
            ans = list(zip(a.get("text", []), a.get("answer_start", [])))
            add(items, lang, split, r["question"], r["context"], ans)
    for r in json.load(open(os.path.join(raw, "fido_train.json")))["data"]:
        add(items, "uk", "train", r["question"], r["context"], [(r["answer"], None)])
    arts = json.load(open(os.path.join(raw, "fido_val.json")))["data"]
    rng = random.Random(5)
    rng.shuffle(arts)
    cut = int(len(arts) * 0.75)
    for ai, art in enumerate(arts):
        split = "train" if ai < cut else "val"
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                ans = [] if qa.get("is_impossible") else [(x["text"], x["answer_start"]) for x in qa["answers"]][:1]
                add(items, "uk", split, qa["question"], p["context"], ans)
    for r in pq.read_table(os.path.join(raw, "xquad_ru.parquet")).to_pylist():
        a = r["answers"]
        add(items, "ru", "test_xquad", r["question"], r["context"], list(zip(a["text"], a["answer_start"])))
    return items


def train_vocab(items, size):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    rng = random.Random(1)
    per_lang = {"en": [], "ru": [], "uk": []}
    seen = set()
    for it in items:
        if it["split"] != "train":
            continue
        per_lang[it["lang"]].append(it["q"])
        if it["ctx"] not in seen:
            seen.add(it["ctx"])
            per_lang[it["lang"]].append(it["ctx"])
    budget = min(sum(len(t) for t in v) for v in per_lang.values())
    corpus = []
    for lang, texts in per_lang.items():
        rng.shuffle(texts)
        used = 0
        for t in texts:
            if used > budget:
                break
            corpus.append(" ".join(w.lower() for w, _, _ in R.words(t)))
            used += len(t)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=100))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.WordPieceTrainer(vocab_size=size, min_frequency=3, special_tokens=["[UNK]", "[PAD]"],
                                        continuing_subword_prefix=R.CONT, limit_alphabet=400)
    tok.train_from_iterator(corpus, trainer)
    v = tok.get_vocab()
    toks = [t for t, _ in sorted(v.items(), key=lambda kv: kv[1]) if t not in ("[UNK]", "[PAD]")]
    toks = [t for t in toks if len(t[len(R.CONT):] if t.startswith(R.CONT) else t) <= R.MAX_TOKEN]
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for t in toks:
        vocab.setdefault(t, len(vocab))
    print(f"vocabulary: {len(vocab)} tokens (corpus {len(corpus)} texts, {budget} chars per language)")
    return vocab


def example(it, vocab, ctx=None):
    """arrays of one example; ctx overrides the paragraph (distractors)"""
    text = ctx if ctx is not None else it["ctx"]
    sel = R.select_context(text, it["q"], vocab)
    toks = sel["toks"]
    feats = R.features(sel, it["q"])
    ws = sel["words"]
    start = end = -1
    if ctx is None and it["spans"]:
        a, b = it["spans"][0]
        inside = [ti for ti, (_, wi) in enumerate(toks) if ws[wi][1] >= a and ws[wi][2] <= b]
        if inside:
            w_first = toks[inside[0]][1]
            w_last = toks[inside[-1]][1]
            # the whole answer must be in the window: its first and last word, all tokens
            all_first = ws[w_first][1] <= a + 1
            all_last = ws[w_last][2] >= b - 1
            if all_first and all_last and inside == list(range(inside[0], inside[-1] + 1)):
                start, end = inside[0], inside[-1]
    n = len(toks)
    ids = np.full(R.MAXC, 1, np.int32)
    ids[:n] = [t for t, _ in toks]
    f = np.zeros((R.MAXC, 4), np.float32)
    f[:n] = feats
    ws_flag = np.zeros(R.MAXC, np.int8)
    we_flag = np.zeros(R.MAXC, np.int8)
    for ti, (_, wi) in enumerate(toks):
        if ti == 0 or toks[ti - 1][1] != wi:
            ws_flag[ti] = 1
        if ti == n - 1 or toks[ti + 1][1] != wi:
            we_flag[ti] = 1
    q = R.question_tokens(it["q"], vocab)
    qids = np.full(R.MAXQ, 1, np.int32)
    qids[:len(q)] = q
    return ids, n, f, ws_flag, we_flag, qids, len(q), start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--distractors", type=float, default=0.2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = load(a.raw)
    print("loaded", len(items), "questions")
    vocab = train_vocab(items, a.vocab)
    json.dump(vocab, open(os.path.join(a.out, "vocab.json"), "w"), ensure_ascii=False)
    rng = random.Random(7)
    by_lang = {}
    for it in items:
        by_lang.setdefault((it["lang"], it["split"]), []).append(it["ctx"])
    cols = {k: [] for k in ("ids", "n", "f", "ws", "we", "q", "qn", "s", "e", "lang", "split", "kind")}
    langs = {"en": 0, "ru": 1, "uk": 2}
    splits = {"train": 0, "val": 1, "test_xquad": 2}

    def push(ex, it, kind):
        for k, v in zip(("ids", "n", "f", "ws", "we", "q", "qn", "s", "e"), ex):
            cols[k].append(v)
        cols["lang"].append(langs[it["lang"]])
        cols["split"].append(splits[it["split"]])
        cols["kind"].append(kind)

    for i, it in enumerate(items):
        ex = example(it, vocab)
        kind = 0 if it["impossible"] else (1 if ex[7] >= 0 else 2)     # 0 unanswerable, 1 answer in window, 2 retrieval missed
        push(ex, it, kind)
        if not it["impossible"] and rng.random() < a.distractors:
            pool = by_lang[(it["lang"], it["split"])]
            other = pool[rng.randrange(len(pool))]
            if other != it["ctx"]:
                push(example(it, vocab, ctx=other), it, 3)                # 3 unrelated paragraph
        if i % 20000 == 0:
            print(f"  {i}/{len(items)}", flush=True)
    arr = {k: np.array(v) for k, v in cols.items()}
    np.savez_compressed(os.path.join(a.out, "reader_data.npz"), **arr)
    kinds = np.bincount(arr["kind"], minlength=4)
    print("examples", len(arr["s"]), "| unanswerable", kinds[0], "answer in window", kinds[1], "retrieval missed", kinds[2], "unrelated", kinds[3])
    for ln, li in langs.items():
        m = (arr["lang"] == li) & (arr["split"] == 0)
        print(f"  {ln}: train {int(m.sum())}, with answer {int((m & (arr['s'] >= 0)).sum())}")
    unk = (arr["ids"] == 0).sum() / max(1, (arr["ids"] != 1).sum())
    print(f"unknown tokens: {unk * 100:.2f}%   mean context tokens {arr['n'].mean():.1f}, question {arr['qn'].mean():.1f}")


if __name__ == "__main__":
    main()
