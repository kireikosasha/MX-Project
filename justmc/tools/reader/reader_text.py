"""
Text processing of the Millennium reader, shared by the dataset builder and (rule by rule) by the
JustMC code in demo_reader.jc. Everything here must stay simple enough to be done with JustMC
text actions (regex, lower case, substrings, maps).
"""
import re

MAXC = 56          # context tokens read by the model
MAXQ = 16          # question tokens
MAX_TOKEN = 16     # longest vocabulary entry tried by the greedy tokenizer
PREFIX = 5         # words are matched by their first PREFIX letters (cheap stemming)
CONT = "##"        # prefix of a word-continuation token

# Unicode spaces -> " ", stress marks removed, apostrophes unified (same in demo_reader.jc)
_SPACES = re.compile("[  -​  　\t\r﻿]")
_MARKS = re.compile("[̀́]")
_APOS = re.compile("[’ʼ`´]")
WORD_RE = re.compile(r"(?:[^\W_]|')+|[^\w\s]|_")          # Java: (?:[\p{L}\p{N}]|')+|[^\p{L}\p{N}_\s]|_
# a sentence ends at . ! ? ... unless the dot closes a 1-2 letter abbreviation or an initial ("фр.",
# "г.", "А. С.") - fixed-width look-behinds, valid in Python and Java alike
_AB = "[a-zа-яёіїєґ]"
_IN = "[A-ZА-ЯЁІЇЄҐ]"
_END = "[.!?\u2026]"
# the cut is the run of spaces after the sentence end; nothing is consumed but the spaces, so JustMC can
# replace the match by a marker character (its regex replacement cannot use $1)
SENT_RE = re.compile("(?<![\\s(]" + _AB + _END + ")(?<![\\s(]" + _AB + "{2}" + _END + ")(?<![\\s(]" + _IN + _END + ")"
                     "(?<=" + _END + ")\\s+(?=[A-ZА-ЯЁІЇЄҐ0-9\"«(])")


# brackets of a Wikipedia page that hold no digits (pronunciation, names in other languages, notes) are
# removed before the selection: they eat the 56-token window. Brackets with digits (dates) stay.
# JustMC: the same three regex replacements (innermost brackets, so twice for nested ones).
_BRK = re.compile(r"\s*\[[^\[\]]*\]")
_PAR = re.compile(r"\s*\([^()\d]*\)")


def clean_brackets(t):
    t = _BRK.sub("", t)
    t = _PAR.sub("", t)
    return _PAR.sub("", t)


def normalize(t):
    t = _SPACES.sub(" ", t)
    t = _MARKS.sub("", t)
    t = _APOS.sub("'", t)
    return t


def words(t):
    """[(word, start, end)] in t"""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(t)]


def sentences(t):
    """[(start, end)] of the sentences of t: cut at SENT_RE and at line breaks, parts stripped.
    In JustMC: regex replace SENT_RE and "\\n" by a marker, split by the marker, trim."""
    cuts = [(m.start(), m.end()) for m in SENT_RE.finditer(t)]
    spans, prev = [], 0
    for a, b in cuts + [(len(t), len(t))]:
        off = prev
        for part in t[prev:a].split("\n"):
            ps = part.strip()
            if ps:
                st = off + part.find(ps)
                spans.append((st, st + len(ps)))
            off += len(part) + 1
        prev = b
    return spans


STOP = {
    "en": set("""a an the of in on at to for from by with and or but is are was were be been being do does did
    what which who whom whose when where why how that this these those it its as into about than then there their
    they he she his her him i you we our your my me not no yes can could would should will shall may might must
    has have had also such any some all each other more most many much name named called year years""".split()),
    "ru": set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
    вот от меня еще нет о из ему когда даже ли если уже или ни быть был него до вас там потом себя ничего ей может
    они тут где есть надо ней для мы тебя их чем была сам без чего кто этот того потому этого какой какая какое
    какие каких каким какую каком сколько который которая которое которые которого которой которых зачем почему
    куда откуда при об после над через эти нас про них эту эта это этой перед тот та те той тех том между является
    являлся являлась были году года год годах называется называли назывался также""".split()),
    "uk": set("""і й та в у на з із зі до від що як це не так але за по для про при через чи або коли де хто який
    яка яке які якого якої якій яким якому скільки чому навіщо куди звідки він вона воно вони його її їх ми ви я ти
    був була було були є бути також ще вже лише тільки саме цей ця ці того цього цієї тих року рік роках роки
    називається називали є було між після перед над""".split()),
}
ALL_STOP = STOP["en"] | STOP["ru"] | STOP["uk"]
UK_ONLY = STOP["uk"] - STOP["ru"]
RU_ONLY = STOP["ru"] - STOP["uk"]


def language(t):
    """uk / ru / en: letters only one of the languages has, then the stop words only one of them has
    (ties -> ru); no Cyrillic at all -> en"""
    low = t.lower()
    if not re.search("[а-яёіїєґ]", low):
        return "en"
    if re.search("[іїєґ']", low):
        return "uk"
    if re.search("[ыэъё]", low):
        return "ru"
    ws = [w.lower() for w, _, _ in words(t)]
    uk = sum(1 for w in ws if w in UK_ONLY)
    ru = sum(1 for w in ws if w in RU_ONLY)
    return "uk" if uk > ru else "ru"


def is_key(w):
    """a word that carries meaning: letters or digits, not a stop word"""
    lw = w.lower()
    return bool(re.match(r"^(?:[^\W_]|')+$", lw)) and lw not in ALL_STOP


def keys(ws):
    return {w.lower()[:PREFIX] for w in ws if is_key(w)}


def wordpiece(lw, vocab):
    """greedy longest-match-first split of a lower-case word into vocabulary ids ([UNK] = 0)"""
    out, start = [], 0
    n = len(lw)
    while start < n:
        end = min(n, start + MAX_TOKEN)
        tok = None
        while end > start:
            piece = lw[start:end] if start == 0 else CONT + lw[start:end]
            if piece in vocab:
                tok = vocab[piece]
                break
            end -= 1
        if tok is None:
            return [0]
        out.append(tok)
        start = end
    return out


def tokenize_words(ws, vocab):
    """ws: [(word, start, end)] -> tokens [(id, word_index)]"""
    toks = []
    for wi, (w, s, e) in enumerate(ws):
        for t in wordpiece(w.lower(), vocab):
            toks.append((t, wi))
    return toks


def _toks(ws, vocab):
    return [(t, wi) for wi, (w, _, _) in enumerate(ws) for t in wordpiece(w.lower(), vocab)]


def best_window(toks, ws, qk, size):
    """start of the size-token window with most key-word tokens (earliest on ties)"""
    hit = [1 if (is_key(ws[wi][0]) and ws[wi][0].lower()[:PREFIX] in qk) else 0 for _, wi in toks]
    best_s, run = 0, sum(hit[:size])
    best_v = run
    for st in range(1, len(toks) - size + 1):
        run += hit[st + size - 1] - hit[st - 1]
        if run > best_v:
            best_v, best_s = run, st
    return best_s


def select_context(text, question, vocab):
    """what the JustMC reader reads (MAXC tokens at most):
    the sentence sharing most question keys (ties: the earlier one); if it is too long, its window
    with most key words; otherwise the second best sentence fills the remaining tokens - its first
    whole words when it comes after, its last whole words when it comes before.
    Returns dict(words=[(w, s, e)] in reading order, toks=[(id, word index)], score, qkeys)"""
    qk = keys([w for w, _, _ in words(question)])
    sents = sentences(text)
    if not sents:
        return {"words": [], "toks": [], "score": 0, "qkeys": qk}
    scored = []
    for si, (s, e) in enumerate(sents):
        scored.append((len(qk & keys([w for w, _, _ in words(text[s:e])])), si))
    scored.sort(key=lambda x: (-x[0], x[1]))

    def sent_words(si):
        s, e = sents[si]
        return [(w, s + a, s + b) for w, a, b in words(text[s:e])]

    b = scored[0][1]
    bw = sent_words(b)
    bt = _toks(bw, vocab)
    if len(bt) > MAXC:
        st = best_window(bt, bw, qk, MAXC)
        keep = sorted({wi for _, wi in bt[st:st + MAXC]})
        # whole words only
        keep = [wi for wi in keep if all(st <= x < st + MAXC for x, (_, wj) in enumerate(bt) if wj == wi)]
        out = [bw[wi] for wi in keep]
    else:
        out = bw
        room = MAXC - len(bt)
        if len(scored) > 1 and room > 0:
            s2 = scored[1][1]
            w2 = sent_words(s2)
            t2 = _toks(w2, vocab)
            if s2 > b:
                keep = [wi for wi in sorted({wi for _, wi in t2[:room]})
                        if all(x < room for x, (_, wj) in enumerate(t2) if wj == wi)]
                out = bw + [w2[wi] for wi in keep]
            else:
                lo = len(t2) - room
                keep = [wi for wi in sorted({wi for _, wi in t2[max(0, lo):]})
                        if all(x >= lo for x, (_, wj) in enumerate(t2) if wj == wi)]
                out = [w2[wi] for wi in keep] + bw
    toks = _toks(out, vocab)[:MAXC]
    return {"words": out, "toks": toks, "score": scored[0][0], "qkeys": qk}


def features(sel, question):
    """per context token: [exact key match, prefix key match, has digit, capitalized]"""
    qw = {w.lower() for w, _, _ in words(question) if is_key(w)}
    qk = sel["qkeys"]
    out = []
    for _, wi in sel["toks"]:
        w = sel["words"][wi][0]
        lw = w.lower()
        k = is_key(w)
        out.append([1.0 if (k and lw in qw) else 0.0,
                    1.0 if (k and lw[:PREFIX] in qk) else 0.0,
                    1.0 if re.search(r"\d", w) else 0.0,
                    1.0 if (w[:1] != w[:1].lower()) else 0.0])
    return out


def question_tokens(question, vocab):
    toks = tokenize_words(words(question), vocab)
    return [t for t, _ in toks][:MAXQ]
