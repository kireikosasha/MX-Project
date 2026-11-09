#!/usr/bin/env python3
"""
Training sequences of the Millennium chat model:   [U] message [B] reply [E]
  - dialogue pairs (chat_sources.py, chat_translate.py) in English, Russian and Ukrainian;
  - questions of the QA datasets (SQuAD 2.0, SberQuAD, UA-SQuAD) with the reply [W]: "look it up in
    Wikipedia" - the model itself says when it does not know, the reader of demo 3 then answers;
  - a few lines about itself (who / what it is), repeated.
Tokens: the WordPiece vocabulary of the reader (lower case), plus [U] [B] [E] [W].

    python chat_data.py --pairs pairs --mt pairs_mt --qa ../reader/raw --vocab ../reader/data/vocab.json --out data
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reader"))
import reader_text as R  # noqa: E402

MAX_M = 24          # message tokens (the last ones are kept)
MAX_R = 32          # reply tokens (longer replies are dropped)
SPECIAL = ["[U]", "[B]", "[E]", "[W]"]

SELF = {
    "ru": [
        (["кто ты", "ты кто", "кто ты такой", "что ты такое", "ты бот", "ты ии", "ты кто вообще", "представься"],
         ["я ии на нейросети millennium, живу прямо в этом мире.", "я бот на millennium. маленький, но свой.",
          "нейросеть millennium, работаю прямо в justmc. спрашивай."]),
        (["что ты умеешь", "что умеешь", "чем ты можешь помочь", "что ты можешь"],
         ["болтать и искать ответы в википедии.", "могу поговорить, а если не знаю - найду в википедии."]),
        (["кто тебя создал", "кто тебя сделал", "кто твой создатель"],
         ["меня обучили на библиотеке millennium из проекта mx.", "проект mx, библиотека millennium."]),
        (["привет", "здарова", "хай", "прив", "здравствуй", "добрый день", "салам"],
         ["привет!", "здарова!", "привет, как дела?", "хай!", "привет, чё как?"]),
        (["как дела", "как ты", "как жизнь", "как сам", "чё как", "как дела?"],
         ["нормально, перемножаю матрицы. а у тебя?", "норм. а ты как?", "живу в чате, всё отлично."]),
        (["спасибо", "спс", "благодарю"], ["пожалуйста!", "обращайся.", "не за что."]),
        (["пока", "до свидания", "бывай"], ["пока!", "давай, удачи!", "до связи."]),
        (["как тебя зовут", "как зовут", "твоё имя", "как тебя называть"], ["millennium. можно просто миля.", "меня зовут millennium."]),
        (["ты тупой", "ты глупый", "ты дурак", "тупой бот"], ["во мне два миллиона параметров, чего ты ждал?", "сам такой. я учусь."]),
        (["сколько тебе лет", "ты давно тут"], ["я родился, когда меня загрузили в этот мир.", "пару часов, если считать с обучения."]),
        (["ты живой", "ты человек"], ["нет, я нейросеть прямо в коде justmc.", "я набор чисел, но поболтать могу."]),
    ],
    "uk": [
        (["хто ти", "ти хто", "що ти таке", "ти бот", "представся"],
         ["я ші на нейромережі millennium, живу просто в цьому світі.", "я бот на millennium. маленький, але свій."]),
        (["що ти вмієш", "що вмієш", "чим можеш допомогти"],
         ["балакати й шукати відповіді у вікіпедії.", "можу поговорити, а якщо не знаю - знайду у вікіпедії."]),
        (["хто тебе створив", "хто тебе зробив"], ["мене навчили на бібліотеці millennium з проєкту mx."]),
        (["привіт", "здоров", "вітаю", "добрий день", "хай"], ["привіт!", "здоров!", "привіт, як справи?"]),
        (["як справи", "як ти", "як життя", "як справи?"], ["нормально, множу матриці. а в тебе?", "добре. а ти як?"]),
        (["дякую", "дякс"], ["будь ласка!", "звертайся."]),
        (["бувай", "до побачення", "па"], ["бувай!", "до зв'язку."]),
        (["як тебе звати", "твоє ім'я"], ["millennium. можна просто міля.", "мене звати millennium."]),
        (["ти тупий", "ти дурний"], ["в мені два мільйони параметрів, чого ти чекав?", "сам такий. я вчуся."]),
        (["ти живий", "ти людина"], ["ні, я нейромережа просто в коді justmc."]),
    ],
    "en": [
        (["who are you", "what are you", "are you a bot", "are you an ai", "introduce yourself"],
         ["i'm an ai on the millennium network, living right in this world.", "a little bot on millennium. ask me anything."]),
        (["what can you do", "how can you help"],
         ["chat, and look answers up in wikipedia.", "i can talk, and if i don't know, i check wikipedia."]),
        (["who made you", "who created you"], ["i was trained with the millennium library of the mx project."]),
        (["hi", "hello", "hey", "yo", "good morning"], ["hi!", "hello!", "hey, how are you?"]),
        (["how are you", "how's it going", "what's up", "how are you?"], ["fine, multiplying matrices. you?", "good, and you?"]),
        (["thanks", "thank you", "thx"], ["you're welcome!", "any time."]),
        (["bye", "goodbye", "see you"], ["bye!", "see you!"]),
        (["what is your name", "what's your name"], ["millennium. you can call me milly.", "i'm millennium."]),
        (["you are stupid", "dumb bot"], ["i have two million parameters, what did you expect?", "i'm learning."]),
        (["are you alive", "are you human"], ["no, i'm a neural network right inside justmc."]),
    ],
}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def qa_questions(qa):
    out = []
    for name, lang in (("squad_train", "en"), ("sber_train", "ru"), ("uahplt_train", "uk")):
        for r in pq.read_table(os.path.join(qa, name + ".parquet"), columns=["question"]).to_pylist():
            out.append((lang, r["question"]))
    for r in json.load(open(os.path.join(qa, "fido_train.json"), encoding="utf-8"))["data"]:
        out.append(("uk", r["question"]))
    return out


class Tok:
    def __init__(self, vocab):
        self.vocab = vocab
        self.n = len(vocab)
        self.special = {s: self.n + i for i, s in enumerate(SPECIAL)}
        self.memo = {}

    def __call__(self, text):
        out = []
        for w, _, _ in R.words(R.normalize(text)):
            lw = w.lower()
            if lw not in self.memo:
                self.memo[lw] = R.wordpiece(lw, self.vocab)
            out += self.memo[lw]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="chat_sources.py output")
    ap.add_argument("--mt", required=True, help="chat_translate.py output (<source>_<lang>.jsonl)")
    ap.add_argument("--qa", required=True, help="raw QA datasets of the reader")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki", type=float, default=0.12, help="share of [W] sequences")
    ap.add_argument("--self-reps", type=int, default=40, help="sequences per group of lines about itself")
    ap.add_argument("--sample", type=float, default=1.0, help="share of the dialogue pairs kept (fine-tuning sets)")
    a = ap.parse_args()
    rng = random.Random(9)
    vocab = json.load(open(a.vocab, encoding="utf-8"))
    tok = Tok(vocab)
    S = tok.special
    pairs = []
    for f, rep in (("dailydialog", 1), ("personachat", 1), ("everyday", 3), ("rudialogues", 1), ("aya_uk", 3)):
        pairs += [(r["lang"], r["m"], r["r"]) for r in load_jsonl(os.path.join(a.pairs, f + ".jsonl"))] * rep
    for f, rep in (("everyday_ru", 3), ("everyday_uk", 3), ("dailydialog_ru", 1), ("dailydialog_uk", 1), ("rudialogues_uk", 1)):
        pairs += [(r["lang"], r["m"], r["r"]) for r in load_jsonl(os.path.join(a.mt, f + ".jsonl"))] * rep
    if a.sample < 1:
        rng.shuffle(pairs)
        pairs = pairs[:int(len(pairs) * a.sample)]
    langs = {"en": 0, "ru": 1, "uk": 2}
    seqs, kinds, lang_of = [], [], []
    for lang, m, r in pairs:
        mt, rt = tok(m)[-MAX_M:], tok(r)
        if not mt or not rt or len(rt) > MAX_R or 0 in rt:
            continue
        seqs.append(([S["[U]"]] + mt + [S["[B]"]], rt + [S["[E]"]]))
        kinds.append(0)
        lang_of.append(langs[lang])
    n_chat = len(seqs)
    qs = qa_questions(a.qa)
    rng.shuffle(qs)
    n_w = int(n_chat * a.wiki)
    for lang, q in qs[:n_w]:
        mt = tok(q)[-MAX_M:]
        if mt:
            seqs.append(([S["[U]"]] + mt + [S["[B]"]], [S["[W]"]]))
            kinds.append(1)
            lang_of.append(langs[lang])
    for lang, groups in SELF.items():
        for prompts, replies in groups:
            for _ in range(a.self_reps):
                p = rng.choice(prompts)
                p = p + rng.choice(["", "?", "!", " ?", ")"])
                seqs.append(([S["[U]"]] + tok(p) + [S["[B]"]], tok(rng.choice(replies)) + [S["[E]"]]))
                kinds.append(2)
                lang_of.append(langs[lang])
    L = MAX_M + MAX_R + 3
    X = np.full((len(seqs), L), 1, np.int32)
    M = np.zeros((len(seqs), L), np.int8)          # 1 = predict this token (reply part)
    for i, (m, r) in enumerate(seqs):
        s = m + r
        X[i, :len(s)] = s
        M[i, len(m) - 1:len(s) - 1] = 1             # position t predicts token t + 1
    split = np.array([0 if rng.random() > 0.01 else 1 for _ in seqs], np.int8)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "chat_data.npz"), X=X, M=M, kind=np.array(kinds, np.int8),
                        lang=np.array(lang_of, np.int8), split=split)
    json.dump({"n_vocab": tok.n, "special": S}, open(os.path.join(a.out, "chat_meta.json"), "w"))
    k = np.bincount(kinds, minlength=3)
    print(f"sequences {len(seqs)}: chat {k[0]}, [W] {k[1]}, self {k[2]}; tokens {int((X != 1).sum())}, "
          f"predicted {int(M.sum())}; per language {np.bincount(lang_of, minlength=3).tolist()}")


if __name__ == "__main__":
    main()
