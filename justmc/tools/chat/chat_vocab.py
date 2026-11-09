#!/usr/bin/env python3
"""
WordPiece vocabulary of the chat model, trained on what it reads and writes (dialogues in three languages,
the synthetic data, the questions for Wikipedia). The reader keeps its own vocabulary (Wikipedia text): on
chat messages this one needs ~15% fewer tokens, so replies are shorter to compute.

    python chat_vocab.py --src src --qa raw --out vocab.json --size 19750
The vocabulary used for the published model is in the repository (tools/chat/vocab.json).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_data as C  # noqa: E402
R = C.R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=19750)
    a = ap.parse_args()
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    rng = random.Random(3)
    texts = C.texts_for_vocab(a.src, a.qa, rng)
    per = {"en": [], "ru": [], "uk": []}
    for lang, t in texts:
        per[lang].append(t)
    budget = 2 * min(sum(len(t) for t in v) for v in per.values())      # no language more than twice the smallest
    corpus = []
    for lang, ts in per.items():
        rng.shuffle(ts)
        used = 0
        for t in ts:
            if used > budget:
                break
            corpus.append(" ".join(w.lower() for w, _, _ in R.words(R.normalize(t))))
            used += len(t)
        print(f"  {lang}: {len(ts)} texts, {used} characters used")
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=100))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.WordPieceTrainer(vocab_size=a.size, min_frequency=3, special_tokens=["[UNK]", "[PAD]"],
                                        continuing_subword_prefix=R.CONT, limit_alphabet=300, show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    v = tok.get_vocab()
    toks = [t for t, _ in sorted(v.items(), key=lambda kv: kv[1]) if t not in ("[UNK]", "[PAD]")]
    toks = [t for t in toks if len(t[len(R.CONT):] if t.startswith(R.CONT) else t) <= R.MAX_TOKEN]
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for t in toks:
        vocab.setdefault(t, len(vocab))
    json.dump(vocab, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"vocabulary: {len(vocab)} tokens -> {a.out}")


if __name__ == "__main__":
    main()
