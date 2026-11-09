#!/usr/bin/env python3
"""
Training sequences of the Millennium chat model:
    [U] message [B] reply [E]                               one exchange
    [U] message [B] reply [E] [U] message [B] reply [E]     two exchanges (the model remembers the previous one)
    [U] question [B] [W]                                    "look it up in Wikipedia"

Sources
  - real dialogues (chat_sources.py) and their machine translations (chat_translate.py, optional);
    a reply that repeats too often ("не знаю", "да") is kept only a few times, so the model does not learn
    to answer everything with it;
  - synthetic/: dialogues written for this bot (its character, the game, the server), in three languages:
      *_g*.jsonl  {"lang", "msgs": [...], "replies": [...]}   any message of the group with any reply
      *_d*.jsonl  {"lang", "turns": [m1, r1, m2, r2]}
      *_w*.jsonl  {"lang", "q": "..."}                         questions for Wikipedia
  - questions of the QA datasets (SQuAD 2.0, SberQuAD, UA-SQuAD) and "what is X" templates -> [W];
  - a few fixed lines about itself.
Tokens: the chat WordPiece vocabulary (chat_vocab.py), lower case, plus [U] [B] [E] [W].

    python chat_data.py --src src --qa raw --vocab vocab.json --out data            (training)
    python chat_data.py --src src --qa raw --vocab vocab.json --out data_ft --ft    (fine-tuning: mostly synthetic)
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reader"))
import reader_text as R  # noqa: E402

MAX_M = 24          # message tokens (the last ones are kept)
MAX_R = 28          # reply tokens (longer replies are dropped)
REPLY_CAP = 20      # the same reply of a real dataset is kept at most this many times
SPECIAL = ["[U]", "[B]", "[E]", "[W]"]
SYNTH = os.path.join(HERE, "synthetic")

SELF = {
    "ru": [
        (["кто ты", "ты кто", "кто ты такой", "что ты такое", "ты бот", "ты ии", "ты кто вообще", "представься"],
         ["я миля, ии на нейросети millennium, живу прямо в этом мире.", "я бот на millennium. маленький, но свой.",
          "нейросеть millennium, работаю прямо в коде сервера. спрашивай."]),
        (["что ты умеешь", "что умеешь", "чем ты можешь помочь", "что ты можешь"],
         ["болтать и искать ответы в википедии.", "могу поговорить, а если не знаю - найду в википедии."]),
        (["кто тебя создал", "кто тебя сделал", "кто твой создатель"],
         ["меня обучили на библиотеке millennium из проекта mx.", "проект mx, библиотека millennium."]),
        (["как тебя зовут", "как зовут", "твоё имя", "как тебя называть"], ["millennium. можно просто миля.", "меня зовут миля."]),
        (["ты живой", "ты человек"], ["нет, я нейросеть прямо в коде сервера.", "я набор чисел, но поболтать могу."]),
    ],
    "uk": [
        (["хто ти", "ти хто", "що ти таке", "ти бот", "представся"],
         ["я міля, ші на нейромережі millennium, живу просто в цьому світі.", "я бот на millennium. маленький, але свій."]),
        (["що ти вмієш", "що вмієш", "чим можеш допомогти"],
         ["балакати й шукати відповіді у вікіпедії.", "можу поговорити, а якщо не знаю - знайду у вікіпедії."]),
        (["хто тебе створив", "хто тебе зробив"], ["мене навчили на бібліотеці millennium з проєкту mx."]),
        (["як тебе звати", "твоє ім'я"], ["millennium. можна просто міля.", "мене звати міля."]),
        (["ти живий", "ти людина"], ["ні, я нейромережа просто в коді сервера."]),
    ],
    "en": [
        (["who are you", "what are you", "are you a bot", "are you an ai", "introduce yourself"],
         ["i'm milly, an ai on the millennium network, living right in this world.", "a little bot on millennium. ask me anything."]),
        (["what can you do", "how can you help"],
         ["chat, and look answers up in wikipedia.", "i can talk, and if i don't know, i check wikipedia."]),
        (["who made you", "who created you"], ["i was trained with the millennium library of the mx project."]),
        (["what is your name", "what's your name"], ["millennium. you can call me milly.", "i'm milly."]),
        (["are you alive", "are you human"], ["no, i'm a neural network right inside the server code."]),
    ],
}

# short "what is X" questions also go to Wikipedia (the QA datasets have long questions only)
TEMPLATES = {
    "ru": ["что такое {}", "кто такой {}", "кто такая {}", "где находится {}", "расскажи про {}", "что за {}",
           "кто такие {}", "что ты знаешь про {}", "что такое {} вообще", "а что такое {}"],
    "uk": ["що таке {}", "хто такий {}", "хто така {}", "де знаходиться {}", "розкажи про {}", "що за {}",
           "що ти знаєш про {}", "а що таке {}"],
    "en": ["what is {}", "who is {}", "where is {}", "tell me about {}", "what's {}", "who was {}",
           "what do you know about {}", "what are {}"],
}

CYR = re.compile("[а-яёіїєґ]", re.I)
LAT = re.compile("[a-z]", re.I)
RU_ONLY = re.compile("[ыэъё]", re.I)
UK_ONLY = re.compile("[іїєґ]", re.I)
JUNK = re.compile("[\U00010000-\U0010ffff☀-➿️‍*_#`]")


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except ValueError:
                    pass
    return out


def lang_ok(text, lang):
    """the text is in the language it is filed under (synthetic data check)"""
    cyr, lat = len(CYR.findall(text)), len(LAT.findall(text))
    if lang == "en":
        return cyr == 0
    if cyr == 0 or lat > cyr:
        return lat <= 12 and cyr == 0 and len(text) <= 14        # "ok", "gg", "lol" are fine anywhere
    return not (RU_ONLY.search(text) if lang == "uk" else UK_ONLY.search(text))


def tidy(text):
    return re.sub(r"\s+", " ", JUNK.sub("", str(text))).strip()


def load_synthetic(folder=SYNTH):
    """-> groups [(lang, msgs, replies)], dialogues [(lang, [m1, r1, m2, r2])], facts [(lang, q)]; bad lines dropped"""
    groups, dialogs, facts = [], [], []
    dropped = 0
    seen = set()
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        for r in load_jsonl(path):
            lang = r.get("lang")
            if lang not in ("ru", "uk", "en"):
                dropped += 1
                continue
            if isinstance(r.get("msgs"), list) and isinstance(r.get("replies"), list):
                ms = [tidy(x) for x in r["msgs"] if isinstance(x, str)]
                rs = [tidy(x) for x in r["replies"] if isinstance(x, str)]
                ms = list(dict.fromkeys(m for m in ms if m and len(m.split()) <= 16 and lang_ok(m, lang)))
                rs = list(dict.fromkeys(x for x in rs if x and 1 <= len(x.split()) <= 22 and lang_ok(x, lang) and "[" not in x))
                if len(ms) >= 2 and len(rs) >= 2:
                    groups.append((lang, ms, rs))
                else:
                    dropped += 1
            elif isinstance(r.get("turns"), list) and len(r["turns"]) == 4:
                t = [tidy(x) for x in r["turns"]]
                key = "\n".join(t).lower()
                if all(t) and all(lang_ok(x, lang) for x in t) and all(len(x.split()) <= 22 for x in t) \
                        and "[" not in t[1] + t[3] and key not in seen:
                    seen.add(key)
                    dialogs.append((lang, t))
                else:
                    dropped += 1
            elif isinstance(r.get("q"), str):
                q = tidy(r["q"])
                if q and q.lower() not in seen and len(q.split()) <= 18 and lang_ok(q, lang):
                    seen.add(q.lower())
                    facts.append((lang, q))
                else:
                    dropped += 1
            else:
                dropped += 1
    return groups, dialogs, facts, dropped


def load_real(src):
    """-> dialogs [(lang, src, turns)], pairs [(lang, src, m, r)] (with the translations when they exist)"""
    dialogs, pairs = [], []
    for f in ("dialogs.jsonl", "dialogs_ru.jsonl", "dialogs_uk.jsonl"):
        dialogs += [(r["lang"], r["src"], r["turns"]) for r in load_jsonl(os.path.join(src, f))]
    for f in ("pairs.jsonl", "pairs_uk.jsonl"):
        pairs += [(r["lang"], r["src"], r["m"], r["r"]) for r in load_jsonl(os.path.join(src, f))]
    return dialogs, pairs


def qa_questions(qa):
    out = []
    for name, lang in (("squad_train", "en"), ("sber_train", "ru"), ("uahplt_train", "uk")):
        for r in pq.read_table(os.path.join(qa, name + ".parquet"), columns=["question"]).to_pylist():
            out.append((lang, r["question"]))
    for r in json.load(open(os.path.join(qa, "fido_train.json"), encoding="utf-8"))["data"]:
        out.append(("uk", r["question"]))
    return out


def entities(qa):
    """names for the templates: SQuAD article titles, short capitalised answers of the QA datasets"""
    out = {"en": set(), "ru": set(), "uk": set()}
    t = pq.read_table(os.path.join(qa, "squad_train.parquet"), columns=["title"]).column("title").to_pylist()
    out["en"].update(x.replace("_", " ") for x in set(t))
    for name, lang in (("squad_train", "en"), ("sber_train", "ru"), ("uahplt_train", "uk")):
        for r in pq.read_table(os.path.join(qa, name + ".parquet"), columns=["answers"]).to_pylist():
            for x in r["answers"]["text"][:1]:
                if 1 <= len(x.split()) <= 3 and len(x) <= 30 and x[:1].isupper() and not any(ch.isdigit() for ch in x):
                    out[lang].add(x.strip(" .,"))
    return {k: sorted(v) for k, v in out.items()}


def wiki_templates(qa, rng, n_per_lang):
    ents = entities(qa)
    out = []
    for lang, names in ents.items():
        for _ in range(n_per_lang):
            q = rng.choice(TEMPLATES[lang]).format(rng.choice(names))
            q = q.lower() if rng.random() < 0.6 else q
            if rng.random() < 0.5:
                q += "?"
            out.append((lang, q))
    return out


def reply_key(r):
    return re.sub(r"[\W_]+", " ", r.lower()).strip()


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


class Builder:
    """exchanges -> token sequences with the mask of predicted positions"""

    def __init__(self, tok):
        self.tok, self.S = tok, tok.special
        self.seqs, self.kinds, self.langs = [], [], []

    def exchange(self, m, r):
        mt, rt = self.tok(m)[-MAX_M:], self.tok(r)
        if not mt or not rt or len(rt) > MAX_R or 0 in rt:
            return None
        return [self.S["[U]"]] + mt + [self.S["[B]"]], rt + [self.S["[E]"]]

    def question(self, q):
        mt = self.tok(q)[-MAX_M:]
        return ([self.S["[U]"]] + mt + [self.S["[B]"]], [self.S["[W]"]]) if mt else None

    def add(self, parts, kind, lang):
        """parts: [(context tokens, predicted tokens)] or None items -> one sequence (False if a part is unusable)"""
        if any(p is None for p in parts):
            return False
        toks, tgt = [], []
        for ctx, pred in parts:
            toks += ctx + pred
            tgt += [0] * len(ctx) + [1] * len(pred)
        self.seqs.append((toks, tgt))
        self.kinds.append(kind)
        self.langs.append({"en": 0, "ru": 1, "uk": 2}[lang])
        return True

    def arrays(self, rng, val=0.01):
        L = max(len(t) for t, _ in self.seqs)
        X = np.full((len(self.seqs), L), 1, np.int32)
        M = np.zeros((len(self.seqs), L), np.int8)          # 1 = position t predicts token t + 1
        for i, (toks, tgt) in enumerate(self.seqs):
            X[i, :len(toks)] = toks
            M[i, :len(toks) - 1] = tgt[1:]
        split = (np.array([rng.random() for _ in self.seqs]) < val).astype(np.int8)
        return X, M, split


def texts_for_vocab(src, qa, rng):
    """(lang, text) of everything the model reads or writes, for chat_vocab.py"""
    dialogs, pairs = load_real(src)
    out = [(lang, t) for lang, _, turns in dialogs for t in turns]
    out += [(lang, t) for lang, _, m, r in pairs for t in (m, r)]
    groups, sd, facts, _ = load_synthetic()
    for _ in range(4):                                       # the bot's own words must be whole tokens
        out += [(lang, t) for lang, ms, rs in groups for t in ms + rs]
        out += [(lang, t) for lang, turns in sd for t in turns]
    qs = qa_questions(qa)
    rng.shuffle(qs)
    out += qs[:120000] + facts * 3 + wiki_templates(qa, rng, 8000)
    out += [(lang, t) for lang, gs in SELF.items() for ps, rs in gs for t in ps + rs] * 20
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="chat_sources.py / chat_translate.py output")
    ap.add_argument("--qa", required=True, help="raw QA datasets (download.py)")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft", action="store_true", help="fine-tuning set: all synthetic dialogues, a sample of the rest")
    a = ap.parse_args()
    rng = random.Random(9)
    tok = Tok(json.load(open(a.vocab, encoding="utf-8")))
    B = Builder(tok)
    real = 0.12 if a.ft else 1.0
    dialogs, pairs = load_real(a.src)
    # ---- real pairs, frequent replies capped
    singles = [(lang, m, r) for lang, _, m, r in pairs]
    chains = []
    for lang, src, turns in dialogs:
        rate = 0.5 if src.startswith("personachat") else 1.0
        for i in range(len(turns) - 1):
            if rng.random() < rate:
                singles.append((lang, turns[i], turns[i + 1]))
        for i in range(len(turns) - 3):
            if rng.random() < 0.4 * rate:
                chains.append((lang, turns[i:i + 4]))
        if src.startswith("everyday"):                       # user <-> assistant: the most useful ones, twice
            singles += [(lang, turns[i], turns[i + 1]) for i in range(0, len(turns) - 1, 2)]
            chains += [(lang, turns[i:i + 4]) for i in range(0, len(turns) - 3, 2)]
    rng.shuffle(singles)
    count = {}
    for lang, m, r in singles:
        k = reply_key(r)
        count[k] = count.get(k, 0) + 1
        if count[k] <= REPLY_CAP and rng.random() < real:
            B.add([B.exchange(m, r)], 0, lang)
    for lang, t in chains:
        if rng.random() < real:
            B.add([B.exchange(t[0], t[1]), B.exchange(t[2], t[3])], 1, lang)
    # ---- synthetic
    groups, sdialogs, facts, dropped = load_synthetic()
    per_msg = 6 if a.ft else 3
    for lang, ms, rs in groups:
        for m in ms:
            for r in rng.sample(rs, min(per_msg, len(rs))):
                B.add([B.exchange(m, r)], 2, lang)
    for _ in range(8 if a.ft else 4):
        for lang, t in sdialogs:
            B.add([B.exchange(t[0], t[1]), B.exchange(t[2], t[3])], 3, lang)
    # the second message of a synthetic group conversation: any group after any exchange of the same language
    by_lang = {}
    for g in groups:
        by_lang.setdefault(g[0], []).append(g)
    for lang, gs in by_lang.items():
        for _ in range(len(gs) * (10 if a.ft else 6)):
            g1, g2 = rng.choice(gs), rng.choice(gs)
            B.add([B.exchange(rng.choice(g1[1]), rng.choice(g1[2])), B.exchange(rng.choice(g2[1]), rng.choice(g2[2]))], 3, lang)
    # ---- Wikipedia questions
    qs = qa_questions(a.qa)
    rng.shuffle(qs)
    wiki = qs[:int(45000 * (0.3 if a.ft else 1.0))] + wiki_templates(a.qa, rng, 3000 if a.ft else 6000) + facts * (6 if a.ft else 4)
    for lang, q in wiki:
        B.add([B.question(q)], 4, lang)
    for lang, gs in by_lang.items():                         # a question after some small talk
        lq = [q for l2, q in wiki if l2 == lang]
        for _ in range(2000 if a.ft else 4000):
            g = rng.choice(gs)
            B.add([B.exchange(rng.choice(g[1]), rng.choice(g[2])), B.question(rng.choice(lq))], 4, lang)
    # ---- about itself
    for lang, gs in SELF.items():
        for prompts, replies in gs:
            for _ in range(120 if a.ft else 40):
                p = rng.choice(prompts) + rng.choice(["", "?", "!", " ?", ")"])
                B.add([B.exchange(p, rng.choice(replies))], 5, lang)
    X, M, split = B.arrays(rng)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "chat_data.npz"), X=X, M=M, kind=np.array(B.kinds, np.int8),
                        lang=np.array(B.langs, np.int8), split=split)
    json.dump({"n_vocab": tok.n, "special": tok.special}, open(os.path.join(a.out, "chat_meta.json"), "w"))
    k = np.bincount(B.kinds, minlength=6)
    print(f"sequences {len(B.seqs)}: real {k[0]} + chains {k[1]}, synthetic {k[2]} + chains {k[3]}, [W] {k[4]}, self {k[5]}; "
          f"en/ru/uk {np.bincount(B.langs, minlength=3).tolist()}; tokens {int((X != 1).sum())}, predicted {int(M.sum())}; "
          f"synthetic: {len(groups)} groups, {len(sdialogs)} dialogues, {len(facts)} questions ({dropped} lines dropped)")


if __name__ == "__main__":
    main()
