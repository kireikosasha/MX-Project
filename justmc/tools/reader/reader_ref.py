#!/usr/bin/env python3
"""
Reference of what demo_reader.jc does in JustMC, step by step, on the exported files
(reader_export.py): text rules, tokenizer, 8-bit embeddings, BiLSTM, decoding and the retold answer.
Used to check the export against the PyTorch model and the JustMC module (in a simulator) against
this file, and to try questions offline:

    python reader_ref.py --weights ../../weights/reader --question "Какая столица Франции?" --text page.txt
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_text as R  # noqa: E402

E = 2.718281828459045
INV_E = 0.36787944117144233
E2 = 7.38905609893065
LN_EPS = 0.00001
ATT_EPS = 0.0000000001
LOW = -1000000.0
MAXLEN = 12           # longest answer in tokens
ROW = 128
NB = 251
HL_A, HL_B = "⁅", "⁆"
MESSAGES = {"uk": "Не знаю — не знайшов надійної відповіді.",
            "ru": "Не знаю — не нашёл надёжного ответа.",
            "en": "I don't know — I found no reliable answer."}


def dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def sigm(x):
    x = min(50.0, max(-50.0, x))
    return math.pow(1.0 + math.pow(INV_E, x), -1)


def tanh(x):
    x = min(30.0, max(-30.0, x))
    return 1.0 - 2.0 * math.pow(math.pow(E2, x) + 1.0, -1)


def load(wdir):
    """the save variables JustMC has after ~ai install"""
    man = json.load(open(os.path.join(wdir, "reader.json"), encoding="utf-8"))["manifest"]
    save, model = {}, {"rows": {}, "ln": {}}
    for f in man:
        obj = json.load(open(os.path.join(wdir, f), encoding="utf-8"))
        if "store" in obj:
            save.update(obj["store"])
        elif "config" in obj:
            model["cfg"] = obj["config"]
            model["attn"] = [float(obj["attnB"])] + obj["attnW"]
        elif "ln" in obj:
            model["ln"].update(obj["ln"])
        elif "rows" in obj:
            for d, g, n, row in obj["rows"]:
                model["rows"][(int(d), g, int(n))] = row
    vocab = {}
    for k, m in save.items():
        if k.startswith("rd.v."):
            vocab.update(m)
        elif k.startswith("rd.c."):
            vocab.update({R.CONT + t: i for t, i in m.items()})
    return save, model, vocab


class Reader:
    def __init__(self, wdir):
        self.save, self.model, self.vocab = load(wdir)
        s = self.save
        self.pd, self.hid, self.rows = int(s["rd.pd"]), int(s["rd.hid"]), int(s["rd.rows"])
        self.cache = {}

    # ---- tokenizer exactly as rd_wordpiece (bucket maps)
    def lookup(self, piece, first):
        c = ord(piece[0]) + (37 * ord(piece[1]) if len(piece) > 1 else 0)
        m = self.save.get(f"rd.{'v' if first else 'c'}.{c % NB}")
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

    # ---- projection from the 8-bit text
    def project(self, tid):
        if tid in self.cache:
            return self.cache[tid]
        s = self.save
        c, k = tid // self.rows, tid % self.rows
        cps = [float(ord(ch)) for ch in s[f"rd.e.{c}"][k * ROW:(k + 1) * ROW]]
        sc = s[f"rd.es.{c}"][k]
        p = []
        for j in range(self.pd):
            raw = dot(s[f"rd.P.{j}"], cps)
            p.append(raw * sc + s["rd.PA"][j] * sc + s["rd.pb"][j])
        self.cache[tid] = p
        return p

    # ---- library LSTM (ml_lstm_forward) and attention pooling (ml_pool_attention)
    def lstm(self, X, d, rev):
        H = self.hid
        rows, ln = self.model["rows"], self.model["ln"][str(d)]
        h, c = [0.0] * H, [0.0] * H
        out = []
        order = range(len(X) - 1, -1, -1) if rev else range(len(X))
        for t in order:
            z = [1.0] + list(X[t]) + h
            gf = [sigm(dot(rows[(d, "f", n)], z)) for n in range(H)]
            gi = [sigm(dot(rows[(d, "i", n)], z)) for n in range(H)]
            go = [sigm(dot(rows[(d, "o", n)], z)) for n in range(H)]
            gg = [tanh(min(50.0, max(-50.0, dot(rows[(d, "c", n)], z)))) for n in range(H)]
            cn = [gf[n] * c[n] + gi[n] * gg[n] for n in range(H)]
            pre = [go[n] * tanh(cn[n]) for n in range(H)]
            inv_h = math.pow(H, -1)
            mean = dot(pre, [1.0] * H) * inv_h
            dl = [p - mean for p in pre]
            var = dot(dl, dl) * inv_h
            inv = math.pow(var + LN_EPS, -0.5)
            h = [ln["g"][n] * (dl[n] * inv) + ln["b"][n] for n in range(H)]
            c = cn
            out.append(h)
        return out[::-1] if rev else out

    def bilstm(self, X):
        fo, bo = self.lstm(X, 0, False), self.lstm(X, 1, True)
        return [f + b for f, b in zip(fo, bo)]

    def pool(self, Hs):
        att = self.model["attn"]
        sc = [dot(att, [1.0] + h) for h in Hs]
        mx = max(sc)
        e = [math.pow(E, s - mx) for s in sc]
        inv = math.pow(sum(e) + ATT_EPS, -1)
        a = [x * inv for x in e]
        return [dot([h[i] for h in Hs], a) for i in range(len(Hs[0]))]

    # ---- the question
    def question(self, q):
        qn = R.normalize(q.replace("¦", "|"))
        ws = [w for w, _, _ in R.words(qn)]
        info = {"text": qn, "lang": R.language(qn), "terms": [], "qk": set(), "qw": set(), "ids": []}
        for w in ws:
            if R.is_key(w):
                lw = w.lower()
                info["qw"].add(lw)
                info["qk"].add(lw[:R.PREFIX])
                if w not in info["terms"] and len(info["terms"]) < 8:
                    info["terms"].append(w)
            info["ids"] += self.wordpiece(w.lower())
        info["ids"] = info["ids"][:R.MAXQ]
        s = self.save
        P = [self.project(t) for t in info["ids"]]
        X = [p + [0.0] * (self.pd + 4) for p in P]
        Hq = self.bilstm(X)
        q = self.pool(Hq)
        z = [1.0] + q
        out = 2 * self.hid
        info.update(P=P, cols=[[p[j] for p in P] for j in range(self.pd)],
                    us=[dot(s[f"rd.Ws.{k}"], z) for k in range(out)],
                    ue=[dot(s[f"rd.We.{k}"], z) for k in range(out)],
                    ns=dot(s["rd.ns"], z), ne=dot(s["rd.ne"], z))
        return info

    # ---- one page
    def select(self, text, qi):
        """rd_select: context words (with sentence / position) and token arrays"""
        t = R.clean_brackets(R.normalize(text))
        sents = [t[a:b] for a, b in R.sentences(t)]
        if not sents:
            return None
        scores = []
        for sn in sents:
            ks = {w.lower()[:R.PREFIX] for w, _, _ in R.words(sn) if R.is_key(w)}
            scores.append(len(qi["qk"] & ks))
        best = max(range(len(sents)), key=lambda i: (scores[i], -i))
        if scores[best] == 0:
            return {"score": 0, "sents": sents}
        rest = [i for i in range(len(sents)) if i != best]
        second = max(rest, key=lambda i: (scores[i], -i)) if rest else -1

        def toks(si):
            ws = [w for w, _, _ in R.words(sents[si])]
            return ws, [self.wordpiece(w.lower()) for w in ws]

        words = []                     # (word, sentence, position, tokens)
        bw, bt = toks(best)
        n = sum(len(x) for x in bt)
        if n > R.MAXC:
            hit, first = [], []
            for w, tt in zip(bw, bt):
                h = 1 if (R.is_key(w) and w.lower()[:R.PREFIX] in qi["qk"]) else 0
                first.append(len(hit))
                hit += [h] * len(tt)
            run = sum(hit[:R.MAXC])
            bestv, st = run, 0
            for s0 in range(1, n - R.MAXC + 1):
                run += hit[s0 + R.MAXC - 1] - hit[s0 - 1]
                if run > bestv:
                    bestv, st = run, s0
            for k, (w, tt) in enumerate(zip(bw, bt)):
                if first[k] >= st and first[k] + len(tt) <= st + R.MAXC:
                    words.append((w, best, k, tt))
        else:
            words = [(w, best, k, tt) for k, (w, tt) in enumerate(zip(bw, bt))]
            room = R.MAXC - n
            if second > -1 and room > 0:
                sw, st_ = toks(second)
                if second > best:
                    used = 0
                    for k, (w, tt) in enumerate(zip(sw, st_)):
                        used += len(tt)
                        if used <= room:
                            words.append((w, second, k, tt))
                else:
                    used, frm = 0, len(sw)
                    for tt in reversed(st_):
                        used += len(tt)
                        if used <= room:
                            frm -= 1
                        else:
                            break
                    words = [(sw[k], second, k, st_[k]) for k in range(frm, len(sw))] + words
        ids, wi, wsf, wef, feats = [], [], [], [], []
        for k, (w, _, _, tt) in enumerate(words):
            lw = w.lower()
            key = R.is_key(w)
            f = [1.0 if (key and lw in qi["qw"]) else 0.0, 1.0 if (key and lw[:R.PREFIX] in qi["qk"]) else 0.0,
                 1.0 if re.search(r"\d", w) else 0.0, 1.0 if w[:1] != w[:1].lower() else 0.0]
            for j, tid in enumerate(tt):
                ids.append(tid)
                wi.append(k)
                wsf.append(1 if j == 0 else 0)
                wef.append(1 if j == len(tt) - 1 else 0)
                feats.append(f)
        return {"score": scores[best], "sents": sents, "words": words, "ids": ids, "wi": wi,
                "ws": wsf, "we": wef, "f": feats}

    def read(self, sel, qi):
        X = []
        for t, tid in enumerate(sel["ids"]):
            p = self.project(tid)
            sc = [dot(p, v) for v in qi["P"]]
            mx = max(sc)
            e = [math.pow(E, s - mx) for s in sc]
            inv = math.pow(dot(e, [1.0] * len(e)), -1)
            a = [x * inv for x in e]
            al = [dot(col, a) for col in qi["cols"]]
            X.append(p + al + sel["f"][t])
        Hc = self.bilstm(X)
        st = [dot(h, qi["us"]) if sel["ws"][t] == 1 else LOW for t, h in enumerate(Hc)]
        en = [dot(h, qi["ue"]) if sel["we"][t] == 1 else LOW for t, h in enumerate(Hc)]

        def softmax(x0, xs):
            mx = max(x0, max(xs))
            e0 = math.pow(E, x0 - mx)
            e = [math.pow(E, v - mx) for v in xs]
            inv = math.pow(e0 + dot(e, [1.0] * len(e)), -1)
            return e0 * inv, [v * inv for v in e]

        ps0, ps = softmax(qi["ns"], st)
        pe0, pe = softmax(qi["ne"], en)
        pn = ps0 * pe0
        best, bs, be = 0.0, -1, -1
        pemax = max(pe)
        T = len(ps)
        for i, psi in enumerate(ps):
            if psi * pemax > best:
                for j in range(i, min(T, i + MAXLEN)):
                    p = psi * pe[j]
                    if p > best:
                        best, bs, be = p, i, j
        return {"best": best, "pn": pn, "bs": bs, "be": be, "st": st, "en": en}

    def compose(self, sel, r):
        words = sel["words"]
        wa, wb = sel["wi"][r["bs"]], sel["wi"][r["be"]]
        si = words[wa][1]
        wb = max(k for k in range(wa, wb + 1) if words[k][1] == si)
        pa, pb = words[wa][2], words[wb][2]
        s = sel["sents"][si]
        at, a, b = 0, 0, 0
        for k, (w, _, _) in enumerate(R.words(s)):
            if k > pb:
                break
            i = s.find(w, at)
            if i < 0:
                i = at
            if k == pa:
                a = i
            at = i + len(w)
            if k == pb:
                b = at
        txt = s[:a] + HL_A + s[a:b] + HL_B + s[b:]
        txt = re.sub(r"\s*\([^()⁅⁆]*\)", "", txt)
        txt = re.sub(r"\s*\([^()⁅⁆]*\)", "", txt)
        txt = re.sub(r"\s*\[[^\[\]⁅⁆]*\]", "", txt)
        txt = re.sub(r"\s+(?=[,.;:])", "", txt)
        return txt.replace(HL_A, "&e").replace(HL_B, "&f"), s[a:b]

    def answer(self, question, extracts, tau=None):
        """extracts: page intros in search order -> dict(text, why, best, pn, answer)"""
        tau = self.save.get("rd.tau0", 0.3) if tau is None else tau
        qi = self.question(question)
        res = {"lang": qi["lang"], "terms": qi["terms"], "pages": []}
        if not qi["terms"]:
            return {**res, "why": "no_keys", "text": MESSAGES[qi["lang"]]}
        for ex in extracts:
            if not ex:
                continue
            ex = ex.replace("¦", "|").replace(HL_A, "").replace(HL_B, "")
            sel = self.select(ex, qi)
            if sel is None or sel["score"] == 0 or not sel["ids"]:
                continue
            r = self.read(sel, qi)
            res["pages"].append({"score": sel["score"], "best": r["best"], "pn": r["pn"], "bs": r["bs"], "be": r["be"]})
            if r["best"] > r["pn"] and r["best"] >= tau:
                txt, ans = self.compose(sel, r)
                # a real answer has a key word the question does not have
                if any(R.is_key(w) and w.lower()[:R.PREFIX] not in qi["qk"] for w, _, _ in R.words(ans)):
                    return {**res, "why": "ok", "text": txt, "answer": ans, "best": r["best"], "pn": r["pn"]}
        return {**res, "why": "unsure", "text": MESSAGES[qi["lang"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--text", nargs="+", required=True, help="files with page intros (search order)")
    ap.add_argument("--tau", type=float)
    a = ap.parse_args()
    rd = Reader(a.weights)
    ex = [open(f, encoding="utf-8").read() for f in a.text]
    print(json.dumps(rd.answer(a.question, ex, a.tau), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
