#!/usr/bin/env python3
"""
Training sequences of the big Millennium chat model (millennium-13m). Everything chat_data.py makes, plus:

  numbers      the numbers of a message are slot tokens [N1] .. [N6]; a number of the reply that is one of
               them is written with its slot (chat_text.py)
  calculation  [U] у меня [N1] стака и ещё [N2] штук, сколько всего [B] [*] [N1] 64 [R1] [+] [R1] [N2] [R2]
               всего [R2] , богач [E]
               the outcome token of a step ([R1], [T], [F]) is computed by the code, not predicted: its
               position has mask 0. Sources: random expressions (2+3*4, "5 умножить на 3"), number questions
               (which is bigger, even / odd, divisible, percent ...), word problems
  synthetic2/  written for this model (logic, game knowledge, explanations, chat, three-exchange dialogues):
      *_p*.jsonl  {"lang", "q": [3 phrasings with {A} {B} ..], "steps": [[op, x, y], ..], "a": [..] or
                   "a_true" / "a_false", "ranges": {"A": [lo, hi], ..}}             word-problem templates
      *_f*.jsonl  {"lang", "kind", "t"}                                             question / reply frames
      *_l* _k* _e* _g*.jsonl  {"lang", "msgs", "replies"}                           groups
      *_n*.jsonl  the same with numbers as {A} {B}: messages with numbers that are NOT to be calculated
      *_t*.jsonl  {"lang", "turns": [m1, r1, m2, r2, m3, r3]}
  three exchanges in a row (the model remembers two earlier ones)

    python chat_data_big.py --src src --qa raw --vocab vocab_big.json --out data [--ft]
    python chat_data_big.py --check         # what the synthetic2 files contain and which lines are unusable
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chat_data as C  # noqa: E402
import chat_text as T  # noqa: E402
R = C.R

MAX_M = 24          # message tokens (the last ones are kept)
MAX_R = 40          # reply tokens with the calculation steps
SYNTH2 = os.path.join(HERE, "synthetic2")
ROLES = "ABCD"
PH = re.compile(r"\{([A-Z][0-9]?)\}")
LANGS = ("ru", "uk", "en")
# numbers that must be whole tokens: the constants of calculation steps are one token each
CONSTANTS = sorted(set(range(0, 101)) | {120, 128, 144, 150, 180, 200, 250, 256, 300, 360, 365, 400, 500, 512, 600, 640,
                                         1000, 1024, 1440, 2000, 3600, 10000})

# ---- built-in frames: the generator works without synthetic2/, the files only add variety
BUILTIN = {
    "ru": {
        "calc_q": ["сколько будет {X}", "{X}", "{X} = ?", "{X}=?", "посчитай {X}", "а {X}?", "чему равно {X}", "{X} сколько будет",
                   "миля, {X}", "реши {X}", "сколько это {X}"],
        "calc_a": ["Будет {R}.", "{R}.", "Получается {R}.", "Это {R}, легко.", "Ответ {R}.", "{R}, я считала по шагам."],
        "big_q": ["что больше {A} или {B}", "какое число больше: {A} или {B}", "{A} или {B} что больше"],
        "big_a": ["Больше {W}.", "{W} больше, чем {L}.", "Конечно {W}."],
        "small_q": ["что меньше {A} или {B}", "какое число меньше: {A} или {B}"],
        "small_a": ["Меньше {W}.", "{W} меньше, чем {L}."],
        "gt_q": ["{A} больше {B}?", "{A} больше чем {B}?", "правда что {A} больше {B}"],
        "gt_yes": ["Да, {A} больше.", "Да, больше."], "gt_no": ["Нет, {A} не больше {B}.", "Нет."],
        "lt_q": ["{A} меньше {B}?", "{A} меньше чем {B}?"],
        "lt_yes": ["Да, {A} меньше.", "Да, меньше."], "lt_no": ["Нет, {A} не меньше {B}.", "Нет."],
        "eq_q": ["{A} равно {B}?", "{A} и {B} это одно и то же число?"],
        "eq_yes": ["Да, они равны.", "Да, одно и то же."], "eq_no": ["Нет, это разные числа.", "Нет, не равны."],
        "even_q": ["{A} чётное?", "число {A} чётное?", "{A} это чётное число?"],
        "even_yes": ["Да, чётное.", "Да, {A} чётное."], "even_no": ["Нет, нечётное.", "Нет, {A} нечётное."],
        "odd_q": ["{A} нечётное?", "число {A} нечётное?"],
        "odd_yes": ["Да, нечётное.", "Да, {A} нечётное."], "odd_no": ["Нет, чётное.", "Нет, {A} чётное."],
        "div_q": ["{A} делится на {B}?", "делится ли {A} на {B}", "{A} делится на {B} без остатка?"],
        "div_yes": ["Да, делится.", "Да, нацело."], "div_no": ["Нет, не делится.", "Нет, остаток {R}."],
        "pct_q": ["сколько будет {A} процентов от {B}", "{A}% от {B}", "{A} процентов от {B} это сколько"],
        "pct_a": ["Это {R}.", "{R}.", "Получается {R}."],
        "half_q": ["половина от {A}", "сколько будет половина {A}", "{A} пополам"], "half_a": ["Половина это {R}.", "{R}."],
        "double_q": ["удвой {A}", "{A} умножить на два", "вдвое больше чем {A}"], "double_a": ["Будет {R}.", "{R}."],
        "sq_q": ["{A} в квадрате", "сколько будет {A} в квадрате", "квадрат числа {A}"], "sq_a": ["{R}.", "В квадрате это {R}."],
        "rem_q": ["остаток от деления {A} на {B}", "какой остаток если {A} разделить на {B}"], "rem_a": ["Остаток {R}.", "{R}."],
        "zero_a": ["На ноль делить нельзя.", "На ноль не делят, даже я.", "Деление на ноль? Не выйдет."],
    },
    "uk": {
        "calc_q": ["скільки буде {X}", "{X}", "{X} = ?", "{X}=?", "порахуй {X}", "а {X}?", "чому дорівнює {X}", "міля, {X}", "розв'яжи {X}"],
        "calc_a": ["Буде {R}.", "{R}.", "Виходить {R}.", "Це {R}, легко.", "Відповідь {R}."],
        "big_q": ["що більше {A} чи {B}", "яке число більше: {A} чи {B}"], "big_a": ["Більше {W}.", "{W} більше, ніж {L}."],
        "small_q": ["що менше {A} чи {B}", "яке число менше: {A} чи {B}"], "small_a": ["Менше {W}.", "{W} менше, ніж {L}."],
        "gt_q": ["{A} більше {B}?", "{A} більше ніж {B}?"], "gt_yes": ["Так, {A} більше.", "Так, більше."], "gt_no": ["Ні, {A} не більше {B}.", "Ні."],
        "lt_q": ["{A} менше {B}?", "{A} менше ніж {B}?"], "lt_yes": ["Так, {A} менше.", "Так, менше."], "lt_no": ["Ні, {A} не менше {B}.", "Ні."],
        "eq_q": ["{A} дорівнює {B}?", "{A} і {B} це одне й те саме число?"], "eq_yes": ["Так, вони рівні."], "eq_no": ["Ні, це різні числа."],
        "even_q": ["{A} парне?", "число {A} парне?"], "even_yes": ["Так, парне."], "even_no": ["Ні, непарне."],
        "odd_q": ["{A} непарне?", "число {A} непарне?"], "odd_yes": ["Так, непарне."], "odd_no": ["Ні, парне."],
        "div_q": ["{A} ділиться на {B}?", "чи ділиться {A} на {B}"], "div_yes": ["Так, ділиться.", "Так, націло."], "div_no": ["Ні, не ділиться.", "Ні, остача {R}."],
        "pct_q": ["скільки буде {A} відсотків від {B}", "{A}% від {B}"], "pct_a": ["Це {R}.", "{R}."],
        "half_q": ["половина від {A}", "{A} навпіл"], "half_a": ["Половина це {R}.", "{R}."],
        "double_q": ["подвій {A}", "{A} помножити на два"], "double_a": ["Буде {R}.", "{R}."],
        "sq_q": ["{A} у квадраті", "квадрат числа {A}"], "sq_a": ["{R}.", "У квадраті це {R}."],
        "rem_q": ["остача від ділення {A} на {B}"], "rem_a": ["Остача {R}.", "{R}."],
        "zero_a": ["На нуль ділити не можна.", "На нуль не ділять, навіть я."],
    },
    "en": {
        "calc_q": ["what is {X}", "{X}", "{X} = ?", "{X}=?", "calculate {X}", "and {X}?", "how much is {X}", "milly, {X}", "solve {X}", "what's {X}"],
        "calc_a": ["That's {R}.", "{R}.", "It makes {R}.", "{R}, easy.", "The answer is {R}."],
        "big_q": ["which is bigger {A} or {B}", "what is greater: {A} or {B}"], "big_a": ["{W} is bigger.", "{W} is greater than {L}."],
        "small_q": ["which is smaller {A} or {B}", "what is less: {A} or {B}"], "small_a": ["{W} is smaller.", "{W} is less than {L}."],
        "gt_q": ["is {A} greater than {B}?", "is {A} bigger than {B}"], "gt_yes": ["Yes, {A} is bigger.", "Yes."], "gt_no": ["No, {A} is not bigger than {B}.", "No."],
        "lt_q": ["is {A} less than {B}?", "is {A} smaller than {B}"], "lt_yes": ["Yes, {A} is smaller.", "Yes."], "lt_no": ["No, {A} is not smaller than {B}.", "No."],
        "eq_q": ["is {A} equal to {B}?", "are {A} and {B} the same number"], "eq_yes": ["Yes, they are equal."], "eq_no": ["No, they are different."],
        "even_q": ["is {A} even?", "is {A} an even number"], "even_yes": ["Yes, it is even."], "even_no": ["No, it is odd."],
        "odd_q": ["is {A} odd?", "is {A} an odd number"], "odd_yes": ["Yes, it is odd."], "odd_no": ["No, it is even."],
        "div_q": ["is {A} divisible by {B}?", "does {B} divide {A}"], "div_yes": ["Yes, it divides evenly."], "div_no": ["No, it does not.", "No, the remainder is {R}."],
        "pct_q": ["what is {A} percent of {B}", "{A}% of {B}"], "pct_a": ["That's {R}.", "{R}."],
        "half_q": ["half of {A}", "what is half of {A}"], "half_a": ["Half is {R}.", "{R}."],
        "double_q": ["double {A}", "{A} times two"], "double_a": ["That's {R}.", "{R}."],
        "sq_q": ["{A} squared", "the square of {A}"], "sq_a": ["{R}.", "Squared it is {R}."],
        "rem_q": ["the remainder of {A} divided by {B}"], "rem_a": ["The remainder is {R}.", "{R}."],
        "zero_a": ["You can't divide by zero.", "Nobody divides by zero, not even me."],
    },
}
# what a kind of frame needs: (steps, kinds of the replies by outcome)
NEED = {"calc_q": "X", "big_q": "AB", "small_q": "AB", "gt_q": "AB", "lt_q": "AB", "eq_q": "AB", "even_q": "A", "odd_q": "A",
        "div_q": "AB", "pct_q": "AB", "half_q": "A", "double_q": "A", "sq_q": "A", "rem_q": "AB"}
MAY = {"calc_a": "R", "big_a": "WL", "small_a": "WL", "gt_yes": "AB", "gt_no": "AB", "lt_yes": "AB", "lt_no": "AB", "eq_yes": "AB",
       "eq_no": "AB", "even_yes": "A", "even_no": "A", "odd_yes": "A", "odd_no": "A", "div_yes": "AB", "div_no": "ABR",
       "pct_a": "ABR", "half_a": "AR", "double_a": "AR", "sq_a": "AR", "rem_a": "ABR", "zero_a": ""}
WORDS = {
    "ru": {"+": ["плюс", "прибавить"], "-": ["минус", "отнять"], "*": ["умножить на", "на", "помножить на"],
           "/": ["разделить на", "делить на", "поделить на"], "^": ["в степени"]},
    "uk": {"+": ["плюс", "додати"], "-": ["мінус", "відняти"], "*": ["помножити на", "на", "множити на"],
           "/": ["поділити на", "ділити на", "розділити на"], "^": ["в степені", "у степені"]},
    "en": {"+": ["plus"], "-": ["minus"], "*": ["times", "multiplied by"], "/": ["divided by", "over"], "^": ["to the power of"]},
}
PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}


# =====================================================================================================
# tokens
# =====================================================================================================
class Tok:
    def __init__(self, vocab):
        self.vocab, self.n = vocab, len(vocab)
        self.special = {s: self.n + i for i, s in enumerate(T.SPECIAL)}
        self.slot = [self.special[s] for s in T.SLOT_TOKENS]
        self.res = [self.special[s] for s in T.RES_TOKENS]
        self.op = {o: self.special[f"[{o}]"] for o in T.OPS}
        self.memo = {}

    def piece(self, lw):
        if lw not in self.memo:
            self.memo[lw] = R.wordpiece(lw, self.vocab)
        return self.memo[lw]

    def message(self, text):
        """-> tokens (numbers as slots), the values of the slots"""
        out, vals = [], []
        for w, _, _ in R.words(T.prepare(R.normalize(text))):
            v = T.number(w)
            if v is not None and len(vals) < T.N_SLOTS:
                out.append(self.slot[len(vals)])
                vals.append(v)
            else:
                out += self.piece(w.lower())
        return out, vals

    def reply(self, text, vals=()):
        """-> tokens; a number that is the value of a slot is written with the slot"""
        out = []
        for w, _, _ in R.words(T.prepare(R.normalize(text))):
            v = T.number(w)
            if v is not None and v in vals:
                out.append(self.slot[list(vals).index(v)])
            elif v is not None and "d" in w:                         # 2d5 -> 2 . 5
                a, b = w.split("d")
                out += self.piece(a) + self.piece(".") + self.piece(b)
            else:
                out += self.piece(w.lower())
        return out

    def const(self, n):
        """the token of a constant of a calculation step, None if it is not one token"""
        p = self.piece(str(int(n)))
        return p[0] if len(p) == 1 and p[0] != 0 else None


class Builder:
    """exchanges -> token sequences with the mask of predicted positions"""

    def __init__(self, tok):
        self.tok, self.S = tok, tok.special
        self.seqs, self.kinds, self.langs = [], [], []
        self.bad = {}

    def exchange(self, m, r):
        mt, vals = self.tok.message(m)
        rt = self.tok.reply(r, vals)
        mt = mt[-MAX_M:]
        if not mt or not rt or len(rt) > MAX_R or 0 in rt:
            return None
        return [self.S["[U]"]] + mt + [self.S["[B]"]], rt + [self.S["[E]"]], None

    def question(self, q):
        mt = self.tok.message(q)[0][-MAX_M:]
        return ([self.S["[U]"]] + mt + [self.S["[B]"]], [self.S["[W]"]], None) if mt else None

    def math(self, q, roles, steps, reply):
        """q: the message with the numbers in it; roles: {role: value}, in the order they stand in q;
        steps: [(op, x, y)] with x, y a role, "R1".., or an int; reply(outcome, results) -> text with {A} {R1} .."""
        mt, vals = self.tok.message(q)
        names = list(roles)
        if len(vals) != len(names) or any(abs(vals[i] - roles[n]) > 1e-9 for i, n in enumerate(names)):
            return None                                             # the frame has digits of its own
        slot = {n: self.tok.slot[i] for i, n in enumerate(names)}
        value = dict(roles)
        pred, mask, results, outcome = [], [], [], None
        for op, x, y in steps:
            ops = []
            for z in (x, y):
                if isinstance(z, str) and z in slot:
                    ops.append((slot[z], value[z]))
                elif isinstance(z, str) and re.fullmatch("R[1-4]", z) and int(z[1]) <= len(results):
                    ops.append((self.tok.res[int(z[1]) - 1], results[int(z[1]) - 1]))
                elif isinstance(z, (int, float)) and not isinstance(z, bool) and z == int(z) and self.tok.const(z) is not None:
                    ops.append((self.tok.const(z), float(z)))
                else:
                    return None
            outcome = T.calc(op, ops[0][1], ops[1][1])
            if outcome[0] == "R":
                if len(results) == T.R_SLOTS:
                    return None
                results.append(outcome[1])
                ot = self.tok.res[len(results) - 1]
            else:
                ot = self.S["[T]"] if outcome[0] == "T" else self.S["[F]"]
            pred += [self.tok.op[op], ops[0][0], ops[1][0], ot]
            mask += [1, 1, 1, 0]
            if outcome[0] != "R":
                break                                               # a comparison or an impossible step ends the steps
        text = reply(outcome, results)
        if text is None:
            return None
        for part in re.split(r"(\{[A-Z][0-9]?\})", text):
            m = PH.fullmatch(part)
            if not m:
                rt = self.tok.reply(part)
            elif m.group(1) in slot:
                rt = [slot[m.group(1)]]
            elif re.fullmatch("R[1-4]", m.group(1)) and int(m.group(1)[1]) <= len(results):
                rt = [self.tok.res[int(m.group(1)[1]) - 1]]
            else:
                return None
            pred += rt
            mask += [1] * len(rt)
        if 0 in pred or len(pred) > MAX_R:
            return None
        return [self.S["[U]"]] + mt[-MAX_M:] + [self.S["[B]"]], pred + [self.S["[E]"]], mask + [1]

    def add(self, parts, kind, lang):
        """parts: [(context tokens, predicted tokens, their mask or None)] -> one sequence (False if a part is None)"""
        if any(p is None for p in parts):
            self.bad[kind] = self.bad.get(kind, 0) + 1
            return False
        toks, tgt = [], []
        for ctx, pred, mask in parts:
            toks += ctx + pred
            tgt += [0] * len(ctx) + (mask or [1] * len(pred))
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


# =====================================================================================================
# synthetic2
# =====================================================================================================
def _texts(xs, lang, lo, hi, digits=False):
    out = []
    for x in xs if isinstance(xs, list) else []:
        if not isinstance(x, str):
            continue
        x = C.tidy(x)
        plain = PH.sub("", x)
        if x and lo <= len(plain.split()) <= hi and C.lang_ok(plain, lang) and "[" not in x and (digits or not re.search("[0-9]", plain)):
            out.append(x)
    return list(dict.fromkeys(out))


def load_synthetic2(folder=SYNTH2):
    """-> {"groups": [(lang, msgs, replies, letter)], "dialogs": [(lang, turns)], "problems": [(lang, tpl)],
           "frames": {lang: {kind: [text]}}, "dropped": {file letter: lines}}"""
    out = {"groups": [], "dialogs": [], "problems": [], "frames": {l: {} for l in LANGS}, "dropped": {}, "lines": {}}
    seen = set()
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        letter = os.path.basename(path).split("_")[-1][:1]
        for r in C.load_jsonl(path):
            lang = r.get("lang") if isinstance(r, dict) else None
            ok = False
            if lang in LANGS and isinstance(r.get("msgs"), list) and isinstance(r.get("replies"), list):
                ms, rs = _texts(r["msgs"], lang, 1, 16, True), _texts(r["replies"], lang, 1, 24, True)
                ok = len(ms) >= 2 and len(rs) >= 2
                if ok:
                    out["groups"].append((lang, ms, rs, letter))
            elif lang in LANGS and isinstance(r.get("turns"), list) and len(r["turns"]) == 6:
                t = [C.tidy(x) for x in r["turns"] if isinstance(x, str)]
                key = "\n".join(t).lower()
                ok = len(t) == 6 and all(t) and all(C.lang_ok(x, lang) and len(x.split()) <= 24 for x in t) \
                    and "[" not in "".join(t) and key not in seen
                if ok:
                    seen.add(key)
                    out["dialogs"].append((lang, t))
            elif lang in LANGS and isinstance(r.get("kind"), str) and isinstance(r.get("t"), str):
                kind, x = r["kind"], C.tidy(r["t"])
                marks = set(PH.findall(x))
                plain = PH.sub("", x)
                ok = bool(x) and not re.search("[0-9]", plain) and C.lang_ok(plain, lang) and "[" not in x and (
                    (kind in NEED and marks == set(NEED[kind])) or (kind in MAY and marks <= set(MAY[kind])))
                if ok and x not in out["frames"][lang].setdefault(kind, []):
                    out["frames"][lang][kind].append(x)
            elif lang in LANGS and isinstance(r.get("q"), list) and isinstance(r.get("steps"), list):
                tpl = problem_template(r, lang)
                ok = tpl is not None
                if ok:
                    out["problems"].append((lang, tpl))
            out["lines"][letter] = out["lines"].get(letter, 0) + 1
            if not ok:
                out["dropped"][letter] = out["dropped"].get(letter, 0) + 1
    return out


def problem_template(r, lang):
    """a checked word-problem template or None"""
    qs = _texts(r.get("q"), lang, 2, 40)
    steps = r.get("steps")
    if not qs or not (1 <= len(steps) <= 3) or not all(isinstance(s, list) and len(s) == 3 and s[0] in T.OPS for s in steps):
        return None
    steps = [[s[0]] + [int(z) if isinstance(z, str) and z.isdigit() else z for z in s[1:]] for s in steps]
    roles = sorted(set(PH.findall(qs[0])))
    if not roles or not set(roles) <= set(ROLES):
        return None
    qs = [q for q in qs if sorted(PH.findall(q)) == roles]             # every role exactly once
    n_res = 0
    for i, (op, x, y) in enumerate(steps):
        for z in (x, y):
            if isinstance(z, str):
                if not (z in roles or (re.fullmatch("R[1-3]", z) and int(z[1]) <= n_res)):
                    return None
            elif isinstance(z, bool) or not isinstance(z, int) or not 0 <= z <= 10000:
                return None
        if op in T.COMPARE:
            if i != len(steps) - 1:
                return None
        else:
            n_res += 1
    compare = steps[-1][0] in T.COMPARE
    allowed = set(roles) | {f"R{i + 1}" for i in range(n_res)}

    def replies(key):
        return [a for a in _texts(r.get(key), lang, 1, 24) if set(PH.findall(a)) <= allowed]
    if compare:
        a = {"T": replies("a_true"), "F": replies("a_false")}
        if not a["T"] or not a["F"]:
            return None
    else:
        a = {"R": [x for x in replies("a") if f"R{n_res}" in PH.findall(x) or n_res == 0]}
        if not a["R"]:
            return None
    rg = r.get("ranges") if isinstance(r.get("ranges"), dict) else {}
    ranges = {}
    for role in roles:
        v = rg.get(role)
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v) and 0 <= v[0] <= v[1] <= 100000:
            ranges[role] = (v[0], v[1])
        else:
            ranges[role] = (1, 20)
    if not qs:
        return None
    return {"q": qs, "steps": [tuple(s) for s in steps], "a": a, "ranges": ranges, "roles": roles}


def draw_problem(tpl, rng):
    """numbers for a template such that every result is a whole number >= 0 -> {role: value} or None"""
    for attempt in range(300):
        val = {r: float(rng.randint(*tpl["ranges"][r])) for r in tpl["roles"]}
        res, ok = [], True
        for op, x, y in tpl["steps"]:
            a, b = [val[z] if z in val else res[int(z[1]) - 1] if isinstance(z, str) else float(z) for z in (x, y)]
            o = T.calc(op, a, b)
            if o[0] == "R":
                res.append(o[1])
                if o[1] < 0 or (o[1] != int(o[1]) and attempt < 250):
                    ok = False
                    break
            elif op not in T.COMPARE:
                ok = False
                break
        if ok:
            return val
    return None


# =====================================================================================================
# numbers and expressions
# =====================================================================================================
def num_text(v, rng):
    if v == int(v):
        return str(int(v))
    return T.fmt(v).replace(".", rng.choice([".", ".", ","]))


def draw_number(rng, small=False):
    r = rng.random()
    if small or r < 0.5:
        return float(rng.randint(0, 20))
    if r < 0.8:
        return float(rng.randint(10, 99))
    if r < 0.93:
        return float(rng.randint(100, 9999))
    if r < 0.97:
        return float(rng.randint(10000, 999999))
    return rng.randint(1, 999) / rng.choice([10.0, 10.0, 100.0])


def expression(rng, lang, n_ops=None):
    """-> text with {A} {B} .., roles in order, steps (post-order: the left operand first), values"""
    n_ops = n_ops or rng.choice([1, 1, 1, 1, 2, 2, 2, 3])
    verbal = rng.random() < 0.22 and n_ops <= 2

    def tree(k):
        if k == 0:
            return None
        op = rng.choices(["+", "-", "*", "/", "^"], [30, 25, 25, 15, 3 if not verbal or k == 1 else 0])[0]
        left = rng.randint(0, k - 1)
        return (op, tree(left), tree(k - 1 - left))
    t = tree(n_ops)
    roles, steps, values = [], [], {}
    spaced = rng.random() < 0.6
    signs = {"+": ["+"], "-": ["-"], "*": ["*", "*", "*", "x", "х", "×"], "/": ["/", "/", "/", ":", "÷"], "^": ["^"]}

    def leaf(small):
        name = ROLES[len(roles)]
        roles.append(name)
        values[name] = draw_number(rng, small)
        return name

    def walk(node, parent=0, right=False, small=False):
        """-> (text, operand)"""
        if node is None:
            name = leaf(small)
            return "{" + name + "}", name
        op, l, r = node
        lt, lo = walk(l, PREC[op], False, small or op == "^")
        rt, ro = walk(r, PREC[op], True, small or op in "^/")
        steps.append((op, lo, ro))
        res = f"R{len(steps)}"
        if verbal:
            text = f"{lt} {rng.choice(WORDS[lang][op])} {rt}"
        else:
            s = rng.choice(signs[op])
            text = f"{lt} {s} {rt}" if spaced or (s in "xх" and rng.random() < 0.5) else f"{lt}{s}{rt}"
            if PREC[op] < parent or (PREC[op] == parent and right) or (op == "^" and parent == 3):
                text = f"({text})"
        return text, res
    if verbal and n_ops == 2:                       # words have no brackets: keep trees that read left to right
        op1 = rng.choice(["+", "-"])
        op2 = rng.choice(["+", "-", "*", "/"])
        t = (op2, (op1, None, None), None) if PREC[op2] == 1 else (op1, None, (op2, None, None))
    text, _ = walk(t)
    if any(op == "^" and not (x in values and y in values) for op, x, y in steps):
        return expression(rng, lang, n_ops)         # a power of plain numbers only
    for op, x, y in steps:                          # exponents stay small, no accidental division by zero
        if op == "^" and y in values:
            values[y] = float(rng.randint(2, 3))
            if x in values:
                values[x] = float(rng.randint(1, 12))
        if op == "/" and y in values and values[y] == 0 and rng.random() < 0.9:
            values[y] = float(rng.randint(1, 12))
    if rng.random() < 0.35 and n_ops == 1 and steps[0][0] == "/" and steps[0][2] in values and values[steps[0][2]] != 0:
        values[steps[0][1]] = values[steps[0][2]] * rng.randint(1, 20)      # divisions that come out even
    return text, roles, steps, values


def fill(text, values, rng):
    return PH.sub(lambda m: num_text(values[m.group(1)], rng) if m.group(1) in values else m.group(0), text)


def casing(q, rng):
    q = q.lower() if rng.random() < 0.85 else q
    return q + (rng.choice(["?", "?", " ?"]) if rng.random() < 0.25 and not q.endswith("?") else "")


class Math:
    """sequences that calculate"""

    def __init__(self, B, frames, rng):
        self.B, self.rng = B, rng
        self.F = {}
        for lang in LANGS:
            self.F[lang] = {k: list(dict.fromkeys(v + frames.get(lang, {}).get(k, []))) for k, v in BUILTIN[lang].items()}

    def pick(self, lang, kind):
        return self.rng.choice(self.F[lang][kind])

    def calc(self, lang):
        rng = self.rng
        text, roles, steps, values = expression(rng, lang)
        q = self.pick(lang, "calc_q").replace("{X}", text)
        if rng.random() < 0.02:                                         # division by zero is answered, not computed
            text, roles, steps, values = "{A} / {B}", ["A", "B"], [("/", "A", "B")], {"A": draw_number(rng), "B": 0.0}
            q = self.pick(lang, "calc_q").replace("{X}", rng.choice(["{A} / {B}", "{A}/{B}", "{A} : {B}"]))

        def reply(outcome, results):
            if outcome[0] == "F":                                       # only a division by zero is talked about
                op, x, y = steps[len(results)]
                zero = op in "/%" and (values.get(y) == 0 or (y[0] == "R" and results[int(y[1]) - 1] == 0))
                return self.pick(lang, "zero_a") if zero else None
            return self.pick(lang, "calc_a").replace("{R}", "{R%d}" % len(results))
        return self.B.math(casing(fill(q, values, rng), rng), {r: values[r] for r in roles}, steps, reply)

    def number_question(self, lang, kind):
        rng = self.rng
        A, Bv = float(rng.randint(1, 200)), float(rng.randint(1, 200))
        yes = rng.random() < 0.5
        q = self.pick(lang, kind + "_q")
        if kind in ("big", "small"):
            while Bv == A:
                Bv = float(rng.randint(1, 200))
            if rng.random() < 0.3:
                A, Bv = draw_number(rng), draw_number(rng)
                if A == Bv:
                    Bv = A + 1
            steps = [(">" if kind == "big" else "<", "A", "B")]

            def reply(o, res):
                w, l = ("A", "B") if o[0] == "T" else ("B", "A")
                return self.pick(lang, kind + "_a").replace("{W}", "{" + w + "}").replace("{L}", "{" + l + "}")
        elif kind in ("gt", "lt", "eq"):
            if kind == "eq" and yes:
                Bv = A
            elif Bv == A:
                Bv = A + rng.randint(1, 9)
            steps = [({"gt": ">", "lt": "<", "eq": "="}[kind], "A", "B")]

            def reply(o, res):
                return self.pick(lang, kind + ("_yes" if o[0] == "T" else "_no"))
        elif kind in ("even", "odd"):
            A = float(rng.randint(0, 9999) if rng.random() < 0.3 else rng.randint(0, 100))
            steps = [("%", "A", 2), ("=", "R1", 0 if kind == "even" else 1)]

            def reply(o, res):
                return self.pick(lang, kind + ("_yes" if o[0] == "T" else "_no"))
        elif kind == "div":
            Bv = float(rng.randint(2, 12))
            A = Bv * rng.randint(1, 40) if yes else float(rng.randint(1, 400))
            steps = [("%", "A", "B"), ("=", "R1", 0)]

            def reply(o, res):
                if o[0] == "T":
                    return self.pick(lang, "div_yes")
                return self.pick(lang, "div_no").replace("{R}", "{R1}")
        elif kind == "pct":
            A = float(rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100, rng.randint(1, 100)]))
            Bv = float(rng.choice([10, 20, 50, 100, 200, 500, 1000, rng.randint(1, 2000)]))
            steps = [("*", "A", "B"), ("/", "R1", 100)]

            def reply(o, res):
                return self.pick(lang, "pct_a").replace("{R}", "{R2}")
        elif kind in ("half", "double", "sq"):
            A = float(rng.randint(1, 60) if kind == "sq" else rng.randint(1, 2000))
            steps = [{"half": ("/", "A", 2), "double": ("*", "A", 2), "sq": ("*", "A", "A")}[kind]]

            def reply(o, res):
                return self.pick(lang, kind + "_a").replace("{R}", "{R1}")
        else:                                                           # rem
            A, Bv = float(rng.randint(1, 500)), float(rng.randint(2, 15))
            steps = [("%", "A", "B")]

            def reply(o, res):
                return self.pick(lang, "rem_a").replace("{R}", "{R1}")
        order = [m for m in PH.findall(q) if m in "AB"]
        values = {"A": A, "B": Bv}
        return self.B.math(casing(fill(q, values, rng), rng), {r: values[r] for r in order}, steps, reply)

    def problem(self, lang, tpl):
        rng = self.rng
        values = draw_problem(tpl, rng)
        if values is None:
            return None
        q = rng.choice(tpl["q"])
        order = PH.findall(q)

        def reply(o, res):
            if o[0] == "F" and "F" not in tpl["a"]:
                return None
            return rng.choice(tpl["a"]["R" if o[0] == "R" else o[0]])
        return self.B.math(casing(fill(q, values, rng), rng), {r: values[r] for r in order}, tpl["steps"], reply)


KINDS = ["big", "small", "gt", "lt", "eq", "even", "odd", "div", "pct", "half", "double", "sq", "rem"]


def vocab_texts(src, qa, rng):
    """(lang, text) for chat_vocab.py: what chat_data.py gives plus the words of synthetic2"""
    out = C.texts_for_vocab(src, qa, rng)
    S2 = load_synthetic2()
    for _ in range(4):
        out += [(lang, PH.sub(" ", t)) for lang, ms, rs, _ in S2["groups"] for t in ms + rs]
        out += [(lang, t) for lang, turns in S2["dialogs"] for t in turns]
        out += [(lang, PH.sub(" ", t)) for lang, tpl in S2["problems"] for t in tpl["q"] + sum(tpl["a"].values(), [])]
        for lang in LANGS:
            fr = dict(BUILTIN[lang])
            for k, v in S2["frames"][lang].items():
                fr[k] = fr.get(k, []) + v
            out += [(lang, PH.sub(" ", t)) for v in fr.values() for t in v]
            out += [(lang, w) for ws in WORDS[lang].values() for w in ws] * 5
    return out


def check():
    S2 = load_synthetic2()
    print("synthetic2:", {k: len(v) for k, v in S2.items() if isinstance(v, list)},
          "| frames", {l: sum(len(v) for v in S2["frames"][l].values()) for l in LANGS})
    print("lines by file letter:", S2["lines"], "| dropped:", S2["dropped"])
    by = {}
    for lang, ms, rs, letter in S2["groups"]:
        by[(lang, letter)] = by.get((lang, letter), 0) + 1
    print("groups:", dict(sorted(by.items())))
    print("problems:", {l: sum(1 for x, _ in S2["problems"] if x == l) for l in LANGS},
          "| dialogues:", {l: sum(1 for x, _ in S2["dialogs"] if x == l) for l in LANGS})
    rng = random.Random(1)
    for lang, tpl in S2["problems"][:3]:
        v = draw_problem(tpl, rng)
        print(" ", lang, fill(tpl["q"][0], v, rng) if v else None, tpl["steps"], list(tpl["a"].values())[0][:1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--qa")
    ap.add_argument("--vocab")
    ap.add_argument("--out")
    ap.add_argument("--ft", action="store_true", help="fine-tuning set: all synthetic dialogues and the calculations, a sample of the rest")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
        return
    rng = random.Random(9)
    tok = Tok(json.load(open(a.vocab, encoding="utf-8")))
    B = Builder(tok)
    real = 0.12 if a.ft else 1.0
    dialogs, pairs = C.load_real(a.src)
    # ---- real pairs, frequent replies capped; chains of two and three exchanges
    singles = [(lang, m, r) for lang, _, m, r in pairs]
    chains, chains3 = [], []
    for lang, src, turns in dialogs:
        rate = 0.5 if src.startswith("personachat") else 1.0
        for i in range(len(turns) - 1):
            if rng.random() < rate:
                singles.append((lang, turns[i], turns[i + 1]))
        for i in range(len(turns) - 3):
            if rng.random() < 0.4 * rate:
                chains.append((lang, turns[i:i + 4]))
        for i in range(len(turns) - 5):
            if rng.random() < 0.15 * rate:
                chains3.append((lang, turns[i:i + 6]))
        if src.startswith("everyday"):                       # user <-> assistant: the most useful ones, twice
            singles += [(lang, turns[i], turns[i + 1]) for i in range(0, len(turns) - 1, 2)]
            chains += [(lang, turns[i:i + 4]) for i in range(0, len(turns) - 3, 2)]
            chains3 += [(lang, turns[i:i + 6]) for i in range(0, len(turns) - 5, 2)]
    rng.shuffle(singles)
    count = {}
    for lang, m, r in singles:
        k = C.reply_key(r)
        count[k] = count.get(k, 0) + 1
        if count[k] <= C.REPLY_CAP and rng.random() < real:
            B.add([B.exchange(m, r)], 0, lang)
    for lang, t in chains:
        if rng.random() < real:
            B.add([B.exchange(t[0], t[1]), B.exchange(t[2], t[3])], 1, lang)
    for lang, t in chains3:
        if rng.random() < real:
            B.add([B.exchange(t[0], t[1]), B.exchange(t[2], t[3]), B.exchange(t[4], t[5])], 1, lang)
    # ---- synthetic: the first set and the set written for this model
    groups, sdialogs, facts, dropped = C.load_synthetic()
    S2 = load_synthetic2()
    groups = [(l, ms, rs) for l, ms, rs in groups] + [(l, ms, rs) for l, ms, rs, _ in S2["groups"]]
    per_msg = 5 if a.ft else 3

    def numbers(m, r):
        """a group with numbers as {A} {B}: random numbers, the same in the message and in the reply"""
        marks = set(PH.findall(m))
        if not marks:
            return (m, r) if "{" not in r else None
        if not set(PH.findall(r)) <= marks:
            return None
        vals = {k: float(rng.randint(1, 100) if rng.random() < 0.8 else rng.randint(100, 5000)) for k in marks}
        return fill(m, vals, rng), fill(r, vals, rng)
    for lang, ms, rs in groups:
        for m in ms:
            for r in rng.sample(rs, min(per_msg, len(rs))):
                for _ in range(3 if "{" in m else 1):
                    mr = numbers(m, r)
                    B.add([B.exchange(*mr) if mr else None], 2, lang)
    for _ in range(8 if a.ft else 4):
        for lang, t in sdialogs:
            B.add([B.exchange(t[0], t[1]), B.exchange(t[2], t[3])], 3, lang)
        for lang, t in S2["dialogs"]:
            B.add([B.exchange(t[0], t[1]), B.exchange(t[2], t[3]), B.exchange(t[4], t[5])], 3, lang)
            B.add([B.exchange(t[0], t[1]), B.exchange(t[2], t[3])], 3, lang)
    by_lang = {}
    for g in groups:
        by_lang.setdefault(g[0], []).append(g)

    def small_talk(lang):
        g = rng.choice(by_lang[lang])
        mr = numbers(rng.choice(g[1]), rng.choice(g[2]))
        return B.exchange(*mr) if mr else None
    for lang, gs in by_lang.items():                         # any group after any exchange of the same language
        for _ in range(len(gs) * (8 if a.ft else 5)):
            parts = [small_talk(lang) for _ in range(3 if rng.random() < 0.3 else 2)]
            B.add(parts, 3, lang)
    # ---- Wikipedia questions
    qs = C.qa_questions(a.qa)
    rng.shuffle(qs)
    wiki = qs[:int(45000 * (0.3 if a.ft else 1.0))] + C.wiki_templates(a.qa, rng, 3000 if a.ft else 6000) + facts * (6 if a.ft else 4)
    for lang, q in wiki:
        B.add([B.question(q)], 4, lang)
    for lang in by_lang:                                     # a question after some small talk
        lq = [q for l2, q in wiki if l2 == lang]
        for _ in range(2000 if a.ft else 4000):
            B.add([small_talk(lang), B.question(rng.choice(lq))], 4, lang)
    # ---- about itself
    for lang, gs in C.SELF.items():
        for prompts, replies in gs:
            for _ in range(120 if a.ft else 40):
                p = rng.choice(prompts) + rng.choice(["", "?", "!", " ?", ")"])
                B.add([B.exchange(p, rng.choice(replies))], 5, lang)
    # ---- calculations
    M = Math(B, S2["frames"], rng)
    n_calc, n_kind, n_prob = (24000, 1300, 30) if a.ft else (30000, 1600, 40)
    probs = {l: [t for x, t in S2["problems"] if x == l] for l in LANGS}
    for lang in LANGS:
        made = []
        for _ in range(n_calc):
            made.append(M.calc(lang))
        for kind in KINDS:
            for _ in range(n_kind):
                made.append(M.number_question(lang, kind))
        for tpl in probs[lang]:
            for _ in range(n_prob):
                made.append(M.problem(lang, tpl))
        for part in made:
            r = rng.random()
            if part is not None and r < 0.12 and lang in by_lang:
                B.add([small_talk(lang), part], 6, lang)             # a calculation after small talk
            elif part is not None and r < 0.2 and lang in by_lang:
                B.add([part, small_talk(lang)], 6, lang)             # and small talk after it
            elif part is not None and r < 0.26:
                other = M.calc(lang) if rng.random() < 0.6 else M.number_question(lang, rng.choice(KINDS))
                B.add([part, other], 6, lang)
            else:
                B.add([part], 6, lang)
    X, Mk, split = B.arrays(rng)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "chat_data.npz"), X=X, M=Mk, kind=np.array(B.kinds, np.int8),
                        lang=np.array(B.langs, np.int8), split=split)
    json.dump({"n_vocab": tok.n, "special": tok.special, "max_m": MAX_M, "big": 1}, open(os.path.join(a.out, "chat_meta.json"), "w"))
    k = np.bincount(B.kinds, minlength=7)
    print(f"sequences {len(B.seqs)}: real {k[0]} + chains {k[1]}, synthetic {k[2]} + chains {k[3]}, [W] {k[4]}, self {k[5]}, "
          f"calculations {k[6]}; en/ru/uk {np.bincount(B.langs, minlength=3).tolist()}; tokens {int((X != 1).sum())}, "
          f"predicted {int(Mk.sum())}; longest {X.shape[1]}")
    print(f"synthetic: {len(groups)} groups, {len(sdialogs)} + {len(S2['dialogs'])} dialogues, {len(facts)} questions, "
          f"{len(S2['problems'])} problem templates, frames {sum(len(v) for l in LANGS for v in S2['frames'][l].values())} "
          f"({dropped} + {sum(S2['dropped'].values())} lines dropped); unusable parts by kind: {B.bad}")


if __name__ == "__main__":
    main()
