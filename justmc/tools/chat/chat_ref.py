#!/usr/bin/env python3
"""
Reference of the chat part of demo_reader.jc (v2) on the exported files (chat_export.py): the tokenizer, the
library LSTM step fed with the code points of the 8-bit embedding, attention over the message, the output row,
class / token logits, top-k sampling, the [W] check and the text of the reply. With temperature 0 (greedy)
the JustMC module must write exactly the same reply.

    python chat_ref.py --chat ../../weights/chat --message "привет, как дела?" "а что ты умеешь?"
(several messages are one conversation: the second one continues the first)
"""
import argparse
import json
import math
import re
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reader"))
import reader_text as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_text as T  # noqa: E402
from reader_ref import dot, sigm, tanh, LN_EPS, E  # noqa: E402

I_RE = re.compile(r"(?<![^\W\d_'])i(?![^\W\d_])")       # the English "i" alone -> "I"
LETTER = re.compile(r"[^\W\d_]")
LOW = -1000000.0
TYPN = 16                         # candidates looked at when a token is drawn
KS_K = 8                          # candidates of the predictability measure
SAMP, TYP, KS = 1, 0.9, 0.75      # defaults: typical sampling, its mass, the "predictable" threshold
REP = 1.0                         # logit penalty per earlier use of a token in the reply
NB = 251
NOSPACE = {".", ",", "!", "?", ";", ":", ")", "»", "%", "…"}
PARTICLES = {"то", "нибудь", "либо", "таки", "ка", "небудь", "но", "бо"}     # что-то, скажи-ка, що-небудь


def lstm_step(rows, ln, H, x, h, c):
    """the library step (ml_lstm_step): gates over z = [1 | x | h], cell, LayerNorm"""
    z = [1.0] + x + h
    gf = [sigm(dot(rows[("f", n)], z)) for n in range(H)]
    gi = [sigm(dot(rows[("i", n)], z)) for n in range(H)]
    go = [sigm(dot(rows[("o", n)], z)) for n in range(H)]
    gg = [tanh(min(50.0, max(-50.0, dot(rows[("c", n)], z)))) for n in range(H)]
    cn = [gf[n] * c[n] + gi[n] * gg[n] for n in range(H)]
    pre = [go[n] * tanh(cn[n]) for n in range(H)]
    inv_h = math.pow(H, -1)
    mean = dot(pre, [1.0] * H) * inv_h
    dl = [p - mean for p in pre]
    var = dot(dl, dl) * inv_h
    inv = math.pow(var + LN_EPS, -0.5)
    return [ln["g"][n] * (dl[n] * inv) + ln["b"][n] for n in range(H)], cn


def load(wdir):
    man = list(json.load(open(os.path.join(wdir, "chat.json"), encoding="utf-8"))["manifest"])
    save, rows, ln = {}, {}, None
    while man:
        obj = json.load(open(os.path.join(wdir, man.pop(0)), encoding="utf-8"))
        if "manifest" in obj:                          # a long list of files comes as a manifest of manifests
            man += obj["manifest"]
        elif "store" in obj:
            save.update(obj["store"])
        elif "ln" in obj:
            ln = obj["ln"]["0"]
        elif "rows" in obj:
            for d, g, n, row in obj["rows"]:
                rows[(g, int(n))] = row
    return save, rows, ln


class Chat:
    def __init__(self, chat_dir):
        self.s, self.rows, self.ln = load(chat_dir)
        s = self.s
        assert s["ch.ver"] in (2, 3)
        self.big = s["ch.ver"] == 3              # numbers as slots, calculation steps (chat_text.py)
        self.D, self.H, self.C, self.R = int(s["ch.dim"]), int(s["ch.hid"]), int(s["ch.C"]), int(s["ch.rows"])
        self.Da, self.Dv = int(s["ch.da"]), int(s["ch.dv"])
        self.sp = {k: int(s[f"ch.{k}"]) for k in "UBEW"}
        self.cls = {k: (int(s[f"ch.c{k}"]), int(s[f"ch.p{k}"])) for k in "EW"}

    # ---- tokenizer (rd_wordpiece with the chat maps)
    def lookup(self, piece, first):
        c = ord(piece[0]) + (37 * ord(piece[1]) if len(piece) > 1 else 0)
        m = self.s.get(f"ch.{'v' if first else 'c'}.{c % NB}")
        return -1 if m is None else m.get(piece, -1)

    def wordpiece(self, lw):
        out, start, n = [], 0, len(lw)
        while start < n:
            end = min(n, start + R.MAX_TOKEN)
            tid = -1
            while end > start:
                tid = self.lookup(lw[start:end], start == 0)
                if tid > -1:
                    break
                end -= 1
            if tid < 0:
                return [0]
            out.append(tid)
            start = end
        return out

    def message_tokens(self, text):
        """-> tokens, the values of the number slots (the big model only)"""
        ids, vals = [], []
        text = R.normalize(text.replace("¦", "|"))
        for w, _, _ in R.words(T.prepare(text) if self.big else text):
            v = T.number(w) if self.big else None
            if v is not None and len(vals) < int(self.s["ch.nslots"]):
                ids.append(int(self.s["ch.n0"]) + len(vals))
                vals.append(v)
            else:
                ids += self.wordpiece(w.lower())
        return ids[-int(self.s["ch.maxm"]):], vals

    def operand(self, tid, vals, results):
        """the value of an operand token of a calculation step, None if it has none"""
        s = self.s
        n0, r0 = int(s["ch.n0"]), int(s["ch.r0"])
        if n0 <= tid < n0 + int(s["ch.nslots"]):
            return vals[tid - n0] if tid - n0 < len(vals) else None
        if r0 <= tid < r0 + int(s["ch.rslots"]):
            return results[tid - r0] if tid - r0 < len(results) else None
        if tid in s["ch.numi"]:
            return float(s["ch.numv"][s["ch.numi"].index(tid)])
        return None

    def shown(self, tid, vals, results):
        """the text of a token in the reply: slots and results are written as their numbers"""
        if self.big:
            s = self.s
            n0, r0 = int(s["ch.n0"]), int(s["ch.r0"])
            if n0 <= tid < n0 + int(s["ch.nslots"]) or r0 <= tid < r0 + int(s["ch.rslots"]):
                v = self.operand(tid, vals, results)
                return "?" if v is None else T.fmt(v)
        return self.text_of(tid)

    # ---- network
    def cps(self, tid):
        c, k = tid // self.R, tid % self.R
        return [float(ord(ch)) for ch in self.s[f"ch.e.{c}"][k * self.D:(k + 1) * self.D]]

    def step(self, x, h, c):
        return lstm_step(self.rows, self.ln, self.H, x, h, c)

    def rows_dot(self, name, n, v):
        return [dot(self.s[f"ch.{name}.{i}"], v) for i in range(n)]

    def attend(self, h, keys, vals):
        if not keys:
            return [0.0] * self.Dv
        q = self.rows_dot("Wq", self.Da, h)
        sc = [dot(q, k) for k in keys]
        mx = max(sc)
        e = [math.pow(E, v - mx) for v in sc]
        tot = 0.0
        for v in e:
            tot += v
        inv = math.pow(tot, -1)
        a = [v * inv for v in e]
        return [dot([v[d] for v in vals], a) for d in range(self.Dv)]

    def out(self, h, x, ctx):
        o = self.rows_dot("Wp", self.D, [1.0] + h + x + ctx)
        return o, dot(o, [1.0] * self.D)

    def class_logits(self, o):
        return self.rows_dot("Wc", self.C, [1.0] + o)

    def word_logits(self, c, o, so):
        s0, soff = self.s["ch.s0"], self.s["ch.so"]
        ids, bs = self.s[f"ch.k.{c}"], self.s[f"ch.kb.{c}"]
        return ids, [dot([dot(self.cps(t), o), so, b], [s0, soff, 1.0]) for t, b in zip(ids, bs)]

    @staticmethod
    def softmax_at(ls, i):
        mx = max(ls)
        tot, ei = 0.0, 0.0
        for k, v in enumerate(ls):
            e = math.pow(E, v - mx)
            tot += e
            if k == i:
                ei = e
        return ei * math.pow(tot, -1)

    def setting(self, name, default):
        """a sampling setting: the user's value (ch.<name>), else the one exported with the model, else the default"""
        s = self.s
        return s.get(f"ch.{name}", s.get(f"ch.{name}0", default))

    def sample(self, ls, k, temp, rng):
        """One token (or class) out of the logits ls -> its index.
        temp 0: the best one. Otherwise the TYPN best candidates are looked at:
          - predictability: the Kolmogorov-Smirnov distance of the best KS_K probabilities from the uniform
            distribution (how the library measures it: max |empirical cdf - reference|); at ks or more the best
            candidate is taken at once - nothing to choose from, and no time spent on choosing;
          - typical sampling (samp 1): the candidates whose information -ln p is closest to the entropy H of the
            distribution, until they cover the mass typ. With ln p = d - ln S (d = logit - best logit) this is
            |d - sum(p d)|: no logarithm needed. One of them is drawn with the weights p^(1/temp);
          - top-k sampling (samp 0): one of the k best with the weights p^(1/temp)."""
        srt = sorted(ls, reverse=True)
        v = srt[0]
        if temp > 0 and len(ls) > 1:
            top = srt[:min(TYPN, len(ls))]
            e = [math.pow(E, x - v) for x in top]
            kk = min(KS_K, len(top))
            inv_k = math.pow(kk, -1)
            tot = 0.0
            for x in e[:kk]:
                tot += x
            inv = math.pow(tot, -1)
            D, cum = 0.0, 0.0
            for j, x in enumerate(e[:kk]):
                cum += x
                D = max(D, cum * inv - (j + 1) * inv_k)
            if D < self.setting("ks", KS):
                inv_t = math.pow(temp, -1)
                if self.setting("samp", SAMP) == 1:
                    vals, w = self.typical_set(top, e, inv_t)
                else:
                    kt = min(k, len(top))
                    vals, w = top[:kt], [math.pow(x, inv_t) for x in e[:kt]]
                tot = 0.0
                for x in w:
                    tot += x
                r = rng.random() * tot
                acc = 0.0
                for x, wx in zip(vals, w):
                    acc += wx
                    v = x
                    if acc >= r:
                        break
        return ls.index(v)

    def typical_set(self, top, e, inv_t):
        tot = 0.0
        for x in e:
            tot += x
        inv = math.pow(tot, -1)
        p = [x * inv for x in e]
        d = [x - top[0] for x in top]
        m = dot(p, d)                                  # = -H - ln S: the mean of ln p up to a constant
        sc = [abs(x - m) for x in d]
        mass = self.setting("typ", TYP)
        vals, w, cum = [], [], 0.0
        for x in sorted(sc):
            if cum >= mass:
                break
            i = sc.index(x)
            sc[i] = 1e9                                # taken
            cum += p[i]
            vals.append(top[i])
            w.append(math.pow(p[i], inv_t))
        return vals, w

    @staticmethod
    def detok(text, t):
        if t.startswith("##"):
            return text + t[2:]
        if not text:
            return t
        if t in PARTICLES and text.endswith(" -"):
            return text[:-2] + "-" + t
        if t in NOSPACE or text.endswith(("(", "«")):
            return text + t
        return text + " " + t

    def text_of(self, tid):
        return self.s[f"ch.t.{tid // 1000}"][tid % 1000]

    def reply(self, message, temp=None, wiki=None, seed=1, state=None, rng=None):
        """state: what the previous reply returned in "state" (the conversation goes on) or None"""
        s = self.s
        temp = s["ch.temp0"] if temp is None else temp
        wiki = s["ch.wiki0"] if wiki is None else wiki
        rng = rng or random.Random(seed)
        if state:                                      # the conversation goes on: ... [E] [U]
            h, c = self.step(self.cps(self.sp["E"]), state["h"], state["c"])
            h, c = self.step(self.cps(self.sp["U"]), h, c)
        else:                                          # the state after [U] from zero, computed at export
            h, c = list(s["ch.h0"]), list(s["ch.c0"])
        msg, nums = self.message_tokens(message)
        keys, vals = [], []
        for t in msg:
            h, c = self.step(self.cps(t), h, c)
            keys.append(self.rows_dot("Wk", self.Da, h))
            vals.append(self.rows_dot("Wv", self.Dv, h))
        x = self.cps(self.sp["B"])
        h, c = self.step(x, h, c)
        ctx = self.attend(h, keys, vals)
        o, so = self.out(h, x, ctx)
        cl = self.class_logits(o)
        cW, pW = self.cls["W"]
        _, wl = self.word_logits(cW, o, so)
        p_w = self.softmax_at(cl, cW) * self.softmax_at(wl, pW)
        res = {"mode": "wiki" if p_w >= wiki else "chat", "sure": p_w >= s["ch.sure0"], "pW": p_w, "tokens": msg}
        out, text = [], ""
        op0 = int(s["ch.op0"]) if self.big else -1
        need, op, args, results, steps = 0, None, [], [], []      # a calculation step in progress, its results
        lp_sum, lp_n = 0.0, 0
        for n in range(int(s["ch.max"])):
            cl = self.class_logits(o)
            cE = self.cls["E"][0]
            if len(s[f"ch.k.{cW}"]) == 1:
                cl[cW] = LOW
            if n == 0 and len(s[f"ch.k.{cE}"]) == 1:
                cl[cE] = LOW
            tt = 0 if need else temp                   # the operands of a step: always the most likely token
            ci = self.sample(cl, int(s["ch.topc"]), tt, rng)
            ids, wl = self.word_logits(ci, o, so)
            for t in out:                              # a token said before is less likely
                if t in ids:
                    wl[ids.index(t)] -= REP
            for t in [self.sp["W"]] + ([self.sp["E"]] if n == 0 else []):
                if t in ids:
                    wl[ids.index(t)] = LOW
            wi = self.sample(wl, int(s["ch.topw"]), tt, rng)
            tid = ids[wi]
            if self.big:                               # confidence: the mean log-probability of the chosen tokens
                lp_sum += math.log(max(1e-12, self.softmax_at(cl, ci) * self.softmax_at(wl, wi)))
                lp_n += 1
            if tid == self.sp["E"]:
                break
            x = self.cps(tid)
            h, c = self.step(x, h, c)
            forced = None
            if need:                                   # an operand of the step
                args.append(self.operand(tid, nums, results))
                need -= 1
                if need == 0:
                    o3 = ("F",) if None in args else T.calc(T.OPS[op], args[0], args[1])
                    if o3[0] == "R" and len(results) >= int(s["ch.rslots"]):
                        o3 = ("F",)
                    if o3[0] == "R":
                        results.append(o3[1])
                        forced = int(s["ch.r0"]) + len(results) - 1
                    else:
                        forced = int(s["ch.T"] if o3[0] == "T" else s["ch.F"])
                    steps.append((T.OPS[op], args[0], args[1], o3))
            elif op0 <= tid < op0 + len(T.OPS) and self.big:
                need, op, args = 2, tid - op0, []
            else:
                out.append(tid)
                text = self.detok(text, self.shown(tid, nums, results))
            if forced is not None:                     # the outcome of the step is fed, not chosen
                x = self.cps(forced)
                h, c = self.step(x, h, c)
            ctx = self.attend(h, keys, vals)
            o, so = self.out(h, x, ctx)
        if self.big and steps and len(LETTER.findall(R.normalize(message.replace("¦", "|")))) < 2:
            o3 = steps[-1][3]                          # no words in the message: the result itself
            text = T.fmt(o3[1]) if o3[0] == "R" else "✔" if o3[0] == "T" else "✘"
        if text:
            text = I_RE.sub("I", text)
            text = text[0].upper() + text[1:]
        res.update(reply=out, text=text, state={"h": h, "c": c}, steps=steps, conf=lp_sum / max(1, lp_n))
        res["unsure"] = self.big and lp_n > 0 and res["conf"] < self.setting("conf", -1000)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--message", nargs="+", required=True)
    ap.add_argument("--temp", type=float)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="every message starts a new conversation")
    a = ap.parse_args()
    ch = Chat(a.chat)
    state = None
    for i, m in enumerate(a.message):
        r = ch.reply(m, a.temp, seed=a.seed, state=None if a.fresh or i % 2 == 0 else state)
        state = r["state"]
        calc = "".join(f" {{{T.fmt(x) if x is not None else '?'} {o} {T.fmt(y) if y is not None else '?'} -> "
                       f"{T.fmt(z[1]) if z[0] == 'R' else z[0]}}}" for o, x, y, z in r["steps"])
        print(f"> {m}\n  [{r['mode']}, P(W) {r['pW']:.2f}, conf {r['conf']:.2f}]{calc} {r['text']}")


if __name__ == "__main__":
    main()
