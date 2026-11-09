#!/usr/bin/env python3
"""
Reference of the chat part of demo_reader.jc on the exported files (chat_export.py): tokens of the message,
the library LSTM step, output vector, class / token logits, top-k sampling, the [W] check and the text of
the reply. With temperature 0 (greedy) the JustMC module must write exactly the same reply.

    python chat_ref.py --chat ../../weights/chat --reader ../../weights/reader --message "привет, как дела?"
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reader"))
import reader_text as R  # noqa: E402
from reader_ref import Reader as ReaderRef, dot, sigm, tanh, LN_EPS, E  # noqa: E402

LOW = -1000000.0
REP = 1.0
OFF = 384.0                       # code point = 256 + code, value = scale * (code - 128)
NOSPACE = {".", ",", "!", "?", ";", ":", ")", "»", "%", "…"}
MAX_M = 24


def load(wdir):
    man = json.load(open(os.path.join(wdir, "chat.json"), encoding="utf-8"))["manifest"]
    save, rows, ln = {}, {}, None
    for f in man:
        obj = json.load(open(os.path.join(wdir, f), encoding="utf-8"))
        if "store" in obj:
            save.update(obj["store"])
        elif "ln" in obj:
            ln = obj["ln"]["0"]
        elif "rows" in obj:
            for d, g, n, row in obj["rows"]:
                rows[(g, int(n))] = row
    return save, rows, ln


class Chat:
    def __init__(self, chat_dir, reader_dir):
        self.s, self.rows, self.ln = load(chat_dir)
        self.tok = ReaderRef(reader_dir)          # vocabulary maps of the reader
        s = self.s
        self.D, self.H, self.C, self.R = int(s["ch.dim"]), int(s["ch.hid"]), int(s["ch.C"]), int(s["ch.rows"])
        self.sp = {k: int(s[f"ch.{k}"]) for k in "UBEW"}
        self.cls = {k: (int(s[f"ch.c{k}"]), int(s[f"ch.p{k}"])) for k in "EW"}

    def message_tokens(self, text):
        ids = []
        for w, _, _ in R.words(R.normalize(text.replace("¦", "|"))):
            ids += self.tok.wordpiece(w.lower())
        return [self.sp["U"]] + ids[-MAX_M:] + [self.sp["B"]]

    def row(self, tid):
        c, k = tid // self.R, tid % self.R
        cps = [float(ord(ch)) for ch in self.s[f"ch.e.{c}"][k * self.D:(k + 1) * self.D]]
        return cps, self.s[f"ch.es.{c}"][k]

    def embed(self, tid):
        cps, sc = self.row(tid)
        off = sc * -OFF
        return [cp * sc + off for cp in cps]

    def step(self, x, h, c):
        H = self.H
        z = [1.0] + x + h
        gf = [sigm(dot(self.rows[("f", n)], z)) for n in range(H)]
        gi = [sigm(dot(self.rows[("i", n)], z)) for n in range(H)]
        go = [sigm(dot(self.rows[("o", n)], z)) for n in range(H)]
        gg = [tanh(min(50.0, max(-50.0, dot(self.rows[("c", n)], z)))) for n in range(H)]
        cn = [gf[n] * c[n] + gi[n] * gg[n] for n in range(H)]
        pre = [go[n] * tanh(cn[n]) for n in range(H)]
        inv_h = math.pow(H, -1)
        mean = dot(pre, [1.0] * H) * inv_h
        dl = [p - mean for p in pre]
        var = dot(dl, dl) * inv_h
        inv = math.pow(var + LN_EPS, -0.5)
        return [self.ln["g"][n] * (dl[n] * inv) + self.ln["b"][n] for n in range(H)], cn

    def out(self, h):
        z = [1.0] + h
        o = [dot(self.s[f"ch.Wp.{j}"], z) for j in range(self.D)]
        return o, dot(o, [1.0] * self.D)

    def class_logits(self, o):
        z = [1.0] + o
        return [dot(self.s[f"ch.Wc.{c}"], z) for c in range(self.C)]

    def word_logits(self, c, o, so, seen=None, mask=()):
        ids, bs = self.s[f"ch.k.{c}"], self.s[f"ch.kb.{c}"]
        out = []
        for tid, b in zip(ids, bs):
            cps, sc = self.row(tid)
            raw = dot(cps, o)
            lg = raw * sc + so * (sc * -OFF) + b
            if seen and tid in seen:
                lg = lg - REP * seen[tid]
            if tid in mask:
                lg = LOW
            out.append(lg)
        return ids, out

    @staticmethod
    def softmax_at(ls, i):
        mx = max(ls)
        e = [math.pow(E, v - mx) for v in ls]
        tot = 0.0
        for v in e:
            tot += v
        return e[i] * math.pow(tot, -1)

    @staticmethod
    def sample(ls, k, temp, rng):
        srt = sorted(ls, reverse=True)
        mx = srt[0]
        best = ls.index(mx)
        if temp <= 0 or k <= 1:
            return best
        thr = srt[min(k, len(ls)) - 1]
        inv = math.pow(temp, -1)
        w, tot = [], 0.0
        for v in ls:
            x = math.pow(E, (v - mx) * inv) if v >= thr else 0.0
            w.append(x)
            tot += x
        r = rng.random() * tot
        acc = 0.0
        for i, x in enumerate(w):
            acc += x
            if acc >= r and x > 0:
                return i
        return best

    @staticmethod
    def detok(text, t):
        if t.startswith("##"):
            return text + t[2:]
        if not text:
            return t
        if t in NOSPACE or text.endswith(("(", "«")):
            return text + t
        return text + " " + t

    def text_of(self, tid):
        return self.s[f"ch.t.{tid // 1000}"][tid % 1000]

    def reply(self, message, temp=None, wiki=None, seed=1):
        s = self.s
        temp = s.get("ch.temp0", 0.8) if temp is None else temp
        wiki = s.get("ch.wiki0", 0.3) if wiki is None else wiki
        rng = random.Random(seed)
        h, c = [0.0] * self.H, [0.0] * self.H
        toks = self.message_tokens(message)
        for t in toks:
            h, c = self.step(self.embed(t), h, c)
        o, so = self.out(h)
        cl = self.class_logits(o)
        cW, pW = self.cls["W"]
        ids, wl = self.word_logits(cW, o, so)
        p_w = self.softmax_at(cl, cW) * self.softmax_at(wl, pW)
        mode = "wiki" if p_w >= wiki else "chat"
        sure = p_w >= s.get("ch.sure0", 0.3)
        out, seen, text = [], {}, ""
        for n in range(int(s["ch.max"])):
            cl = self.class_logits(o)
            for name in ("W", "E"):
                cc, _ = self.cls[name]
                if (name == "W" or n == 0) and len(s[f"ch.k.{cc}"]) == 1:
                    cl[cc] = LOW
            cc = self.sample(cl, int(s["ch.topc"]), temp, rng)
            mask = {self.sp["W"]} | ({self.sp["E"]} if n == 0 else set())
            ids, wl = self.word_logits(cc, o, so, seen, mask)
            tid = ids[self.sample(wl, int(s["ch.topw"]), temp, rng)]
            if tid == self.sp["E"]:
                break
            out.append(tid)
            seen[tid] = seen.get(tid, 0) + 1
            text = self.detok(text, self.text_of(tid))
            h, c = self.step(self.embed(tid), h, c)
            o, so = self.out(h)
        if text:
            text = text[0].upper() + text[1:]
        # mode "wiki": the reader answers first; if it finds nothing, "I don't know" when sure, else this text
        return {"mode": mode, "sure": sure, "pW": p_w, "tokens": toks, "reply": out, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--message", nargs="+", required=True)
    ap.add_argument("--temp", type=float)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    ch = Chat(a.chat, a.reader)
    for m in a.message:
        r = ch.reply(m, a.temp, seed=a.seed)
        print(f"> {m}\n  [{r['mode']}, P(W) {r['pW']:.2f}] {r.get('text', '')}")


if __name__ == "__main__":
    main()
