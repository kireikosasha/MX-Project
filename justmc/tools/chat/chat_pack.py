#!/usr/bin/env python3
"""
The compact files of the big chat model (ch.ver 4).

JustMC keeps the save variables of a world in one file with a size limit ("Variables file is too big"):
the model on 13 M parameters as lists of numbers, with the library copy of its LSTM, was about 70 MB and
did not fit. Here every weight is one character of a text, and nothing is stored twice:

  ch.e.<c>            embeddings, ASCII: 61 levels, code point 96 + code (-30 .. 30), value = se * (cp - 96)
  ch.XL.<g>.<c>       LSTM rows of gate g (f i c o): 1024 characters [x part | h part], U+0180 + code (-127 .. 127)
  ch.XP.<c>           output rows [h | x | context]
  ch.XQ / XK / XV / XC.<c>   attention query / key / value rows and class rows (over 512 values)
  ch.BL.<g>, ch.BP, ch.BC    the biases of those rows, already divided by the scale of the matrix
  <text variable>.h   md5 of every text: demo_reader.jc checks them, a damaged model does not answer

One scale per matrix, and no multiplication by it at run time:
  - the state of the LSTM is kept as K * h (LayerNorm gains are stored times K), so the x part and the h part
    of a row have the same size of weights;
  - a row is the list [bias / q | code points | -384] and its input ends with the sum of the input, so the dot
    product is  bias / q + sum((cp - 384) * input)  = (true value) / q;
  - the gates use the bases e^-q and e^2q, i.e. sigmoid(q * v) = 1 / (1 + (e^-q)^v);
  - the scales that are left are folded into constants (ch.s0, ch.so, ch.qs, ch.cc, ch.qc).
Measured on the published model: 61-level embeddings +0.3 % perplexity, 8-bit matrices +0.2 %.

    python chat_pack.py --src ../../weights/chat13m --out ../../weights/chat13m     (re-pack an exported model)
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reader"))
from reader_export import Packer, dumps, fmt, size, MAX_BYTES, write_manifest  # noqa: E402

E_OFF, E_LEV, E_CLIP = 96, 30, 3.0        # embeddings: ASCII 66 .. 126
M_OFF, M_LEV = 384, 127                   # matrices: U+0101 .. U+01FF
GATES = "fico"


def quant(M, levels, clip=None):
    """-> integer codes, scale (one for the whole matrix)"""
    top = float(np.abs(M).max()) if clip is None else min(float(np.abs(M).max()), clip * float(M.std()))
    q = top / levels
    return np.clip(np.round(M / q), -levels, levels).astype(np.int64), q


def text(codes, off):
    return "".join(map(chr, (codes.reshape(-1) + off).tolist()))


def pack(A):
    """A: the parameters as float arrays (emb, Wx, Wh, b, gam, bet, Wp, Ws, Wa, bp, Wq, Wk, Wv, Wc, bc, bw)
    -> Q: codes, biases and constants of the compact model; Q["eff"]: the float parameters it really computes"""
    H, D = A["Wh"].shape[1], A["emb"].shape[1]
    Q = {}
    ce, se = quant(A["emb"], E_LEV, E_CLIP)
    K = float(A["Wh"].std() / (se * A["Wx"].std()))
    Q.update(e=ce, se=se, K=K, gam=A["gam"] * K, bet=A["bet"] * K)
    eff = {"emb": ce * se}
    # ---- LSTM: rows over [x code points | K h]
    Wx_eff, Wh_eff = np.zeros_like(A["Wx"]), np.zeros_like(A["Wh"])
    for gi, g in enumerate(GATES):
        sl = slice(gi * H, (gi + 1) * H)
        M = np.concatenate([A["Wx"][sl] * se, A["Wh"][sl] / K], 1)
        C, q = quant(M, M_LEV)
        Q[f"L{g}"], Q[f"q{g}"] = C, q
        Q[f"b{g}"] = (A["b"][sl] - E_OFF * q * C[:, :D].sum(1)) / q
        Wx_eff[sl], Wh_eff[sl] = C[:, :D] * q / se, C[:, D:] * q * K
    eff.update(Wx=Wx_eff, Wh=Wh_eff)
    # ---- attention: query / key / value rows over K h
    for name in ("Wq", "Wk", "Wv"):
        C, q = quant(A[name] / K, M_LEV)
        Q[name], Q["q" + name] = C, q
        eff[name] = C * q * K
    # ---- output: rows over [K h | x code points | scaled context]
    hx = np.concatenate([A["Wp"] / K, A["Ws"] * se], 1)
    sc = float(hx.std() / A["Wa"].std())                      # the context is fed divided by sc
    C, qp = quant(np.concatenate([hx, A["Wa"] * sc], 1), M_LEV)
    Q.update(P=C, qp=qp, sc=sc, cc=Q["qWv"] / sc, bP=(A["bp"] - E_OFF * qp * C[:, H:H + D].sum(1)) / qp)
    eff.update(Wp=C[:, :H] * qp * K, Ws=C[:, H:H + D] * qp / se, Wa=C[:, H + D:] * qp / sc)
    # ---- classes: rows over the output divided by qp
    C, q = quant(A["Wc"] * qp, M_LEV)
    Q.update(C=C, qC=q, bC=A["bc"] / q)                       # (capital letters: "qc" is the scale of gate c)
    eff["Wc"] = C * q / qp
    for k in ("b", "gam", "bet", "bp", "bc", "bw"):
        eff[k] = A[k]
    Q["eff"] = eff
    return Q


def lstm_step_q(S, x, h, c):
    """one step of the compact LSTM. S: the store (as loaded by chat_ref), x: code points of the token,
    h: K * state -> (K * new state, new cell). The operations and their order are those of demo_reader.jc."""
    from reader_ref import dot, tanh, LN_EPS
    H = int(S["ch.hid"])
    rows = S["_rows"]
    z = [1.0] + x + h + [dot(x, [1.0] * len(x)) + dot(h, [1.0] * H)]
    g = {}
    for k in "fio":
        B, X = S[f"ch.B{k}"], S[f"ch.X{k}"]
        g[k] = [math.pow(1.0 + math.pow(B, min(X, max(-X, dot(rows[k][n], z)))), -1) for n in range(H)]
    B, X = S["ch.Bc"], S["ch.Xc"]
    gg = [1.0 - 2.0 * math.pow(math.pow(B, min(X, max(-X, dot(rows["c"][n], z)))) + 1.0, -1) for n in range(H)]
    cn = [g["f"][n] * c[n] + g["i"][n] * gg[n] for n in range(H)]
    pre = [g["o"][n] * tanh(cn[n]) for n in range(H)]
    inv_h = math.pow(H, -1)
    mean = dot(pre, [1.0] * H) * inv_h
    dl = [p - mean for p in pre]
    var = dot(dl, dl) * inv_h
    inv = math.pow(var + LN_EPS, -0.5)
    return [S["ch.gam"][n] * (dl[n] * inv) + S["ch.bet"][n] for n in range(H)], cn


def decode(S):
    """the rows of the matrices as the lists demo_reader.jc builds: [bias | code points | -384]"""
    def rows(prefix, n_rows, per, width, bias=None):
        out = []
        for c in range((n_rows + per - 1) // per):
            t = S[f"{prefix}.{c}"]
            for r in range(len(t) // width):
                cp = [float(ord(ch)) for ch in t[r * width:(r + 1) * width]]
                out.append(([float(bias[len(out)])] if bias is not None else []) + cp + [-float(M_OFF)])
        assert len(out) == n_rows, (prefix, len(out), n_rows)
        return out
    H, D, Da, Dv, C = (int(S[k]) for k in ("ch.hid", "ch.dim", "ch.da", "ch.dv", "ch.C"))
    R = {g: rows(f"ch.XL.{g}", H, int(S["ch.nL"]), D + H, S[f"ch.BL.{g}"]) for g in GATES}
    R["P"] = rows("ch.XP", D, int(S["ch.nP"]), H + D + Dv, S["ch.BP"])
    R["Q"] = rows("ch.XQ", Da, int(S["ch.nA"]), H)
    R["K"] = rows("ch.XK", Da, int(S["ch.nA"]), H)
    R["V"] = rows("ch.XV", Dv, int(S["ch.nA"]), H)
    R["C"] = rows("ch.XC", C, int(S["ch.nA"]), D, S["ch.BC"])
    return R


class QPacker(Packer):
    """one text variable per file, with its md5 next to it"""

    def texts(self, prefix, name, rows_text, per):
        """rows_text: the text of every row -> variables <name>.<c> of `per` rows, files <prefix>-NNN.json"""
        n = 0
        for c in range(0, len(rows_text), per):
            t = "".join(rows_text[c:c + per])
            var = f"{name}.{c // per}"
            self.put(f"{prefix}-{n:03d}.json", dumps({"store": {var: t, var + ".h": hashlib.md5(t.encode("utf-8")).hexdigest()}}))
            self.sums.append(var)
            n += 1

    def __init__(self, out):
        super().__init__(out)
        self.sums = []


def per_file(width, bytes_per_char):
    return max(1, (MAX_BYTES - 200) // (width * bytes_per_char))


def write(out, Q, meta, classes, vocab, texts_of, name):
    """the files of the compact model -> number of files. meta: [(variable, value)] of the settings and tokens"""
    import reader_text as R
    from reader_export import bucket
    for f in os.listdir(out) if os.path.isdir(out) else []:
        if f.endswith(".json"):
            os.remove(os.path.join(out, f))
    pk = QPacker(out)
    H, D = Q["Lf"].shape[0], Q["e"].shape[1]
    V = Q["e"].shape[0]
    Dv, Da, C = Q["Wv"].shape[0], Q["Wq"].shape[0], Q["C"].shape[0]
    nE = (MAX_BYTES - 400) // (D + D // 20)                    # a backslash takes two bytes in the file
    nL, nP, nA = per_file(D + H, 2), per_file(H + D + Dv, 2), per_file(max(H, D), 2)
    n = lambda v, d=12: [fmt(x, d) for x in v]                 # noqa: E731
    # the state after [U] from zero, with the compact model itself
    S = {"ch.hid": H, "ch.gam": list(map(float, Q["gam"])), "ch.bet": list(map(float, Q["bet"]))}
    for g in GATES:
        q = Q["q" + g]
        S[f"ch.B{g}"] = math.exp(2 * q) if g == "c" else math.exp(-q)
        S[f"ch.X{g}"] = (30.0 if g == "c" else 50.0) / q
    S["_rows"] = {g: [[float(Q["b" + g][i])] + [float(x + M_OFF) for x in Q["L" + g][i]] + [-float(M_OFF)] for i in range(H)] for g in GATES}
    u = int(dict(meta)["ch.U"])
    h0, c0 = lstm_step_q(S, [float(x + E_OFF) for x in Q["e"][u]], [0.0] * H, [0.0] * H)
    meta = [(k, v) for k, v in meta if k not in ("ch.ver", "ch.rows", "ch.s0", "ch.so", "ch.h0", "ch.c0", "ch.nin")]
    meta = [("ch.ver", 4), ("ch.rows", nE), ("ch.nL", nL), ("ch.nP", nP), ("ch.nA", nA),
            ("ch.s0", fmt(Q["se"] * Q["qp"], 12)), ("ch.so", fmt(-E_OFF * Q["se"] * Q["qp"], 12)),
            ("ch.qs", fmt(Q["qWq"] * Q["qWk"], 12)), ("ch.cc", fmt(Q["cc"], 12)), ("ch.qc", fmt(Q["qC"], 12)),
            ("ch.K", fmt(Q["K"], 12))] + meta
    for g in GATES:
        meta += [(f"ch.B{g}", fmt(S[f"ch.B{g}"], 15)), (f"ch.X{g}", fmt(S[f"ch.X{g}"], 12))]
    pk.store("ch-meta", meta + [("ch.h0", n(h0)), ("ch.c0", n(c0)), ("ch.gam", n(Q["gam"])), ("ch.bet", n(Q["bet"])),
                                ("ch.BP", n(Q["bP"])), ("ch.BC", n(Q["bC"]))] + [(f"ch.BL.{g}", n(Q["b" + g])) for g in GATES])
    net = []
    for c, toks in enumerate(classes):
        net.append((f"ch.k.{c}", toks))
        net.append((f"ch.kb.{c}", [fmt(Q["eff"]["bw"][t], 6) for t in toks]))
    pk.store("ch-net", net)
    pk.store("ch-text", [(f"ch.t.{i}", texts_of[i * 1000:(i + 1) * 1000]) for i in range((V + 999) // 1000)])
    buckets = {}
    for tok, i in vocab.items():
        if tok in ("[UNK]", "[PAD]"):
            continue
        key, kind = (tok[len(R.CONT):], "c") if tok.startswith(R.CONT) else (tok, "v")
        buckets.setdefault(f"ch.{kind}.{bucket(key)}", {})[key] = i
    pk.store("ch-vocab", sorted(buckets.items()))
    for g in GATES:
        pk.texts(f"ch-l{g}", f"ch.XL.{g}", [text(r, M_OFF) for r in Q["L" + g]], nL)
    pk.texts("ch-p", "ch.XP", [text(r, M_OFF) for r in Q["P"]], nP)
    for key, var in (("Wq", "XQ"), ("Wk", "XK"), ("Wv", "XV"), ("C", "XC")):
        pk.texts("ch-" + var.lower(), "ch." + var, [text(r, M_OFF) for r in Q[key]], nA)
    pk.texts("ch-emb", "ch.e", [text(r, E_OFF) for r in Q["e"]], nE)
    pk.put("ch-ready.json", dumps({"store": {"ch.ready": 1}}))
    count = pk.finish("chat")
    base = "https://raw.githubusercontent.com/kireikosasha/MX-Project/master/justmc/weights"
    open(os.path.join(out, "ai.json"), "w").write(dumps({"format": "millennium", "manifest": [
        f"{base}/reader/reader.json", f"{base}/{name}/chat.json"]}))
    stored = sum(len(t.encode("utf-8")) for f in os.listdir(out) for t in [open(os.path.join(out, f), encoding="utf-8").read()])
    return count, stored


def from_export(src):
    """an exported model with number lists (ch.ver 2 / 3) -> A, meta, classes, texts of the tokens, vocabulary"""
    import chat_ref as CR
    s, rows, ln = CR.load(src)
    H, D, V, C = int(s["ch.hid"]), int(s["ch.dim"]), int(s["ch.V"]), int(s["ch.C"])
    Da, Dv, R = int(s["ch.da"]), int(s["ch.dv"]), int(s["ch.rows"])
    s0, OFF = float(s["ch.s0"]), 384.0
    classes = [list(map(int, s[f"ch.k.{c}"])) for c in range(C)]
    cp = np.zeros((V, D))
    for c in range((V + R - 1) // R):
        a = np.frombuffer(s[f"ch.e.{c}"].encode("utf-32-le"), np.uint32).astype(np.float64).reshape(-1, D)
        cp[c * R:c * R + len(a)] = a
    A = {"emb": s0 * (cp - OFF)}
    Wx, Wh, b = np.zeros((4 * H, D)), np.zeros((4 * H, H)), np.zeros(4 * H)
    for gi, g in enumerate(GATES):
        for i in range(H):
            r = np.array(rows[(g, i)], np.float64)
            Wx[gi * H + i], Wh[gi * H + i], b[gi * H + i] = r[1:1 + D] / s0, r[1 + D:], r[0] + OFF * r[1:1 + D].sum()
    A.update(Wx=Wx, Wh=Wh, b=b, gam=np.array(ln["g"]), bet=np.array(ln["b"]))
    Wp, Ws, Wa, bp = np.zeros((D, H)), np.zeros((D, D)), np.zeros((D, Dv)), np.zeros(D)
    for j in range(D):
        r = np.array(s[f"ch.Wp.{j}"], np.float64)
        ws = r[1 + H:1 + H + D]
        Wp[j], Ws[j], Wa[j], bp[j] = r[1:1 + H], ws / s0, r[1 + H + D:], r[0] + OFF * ws.sum()
    A.update(Wp=Wp, Ws=Ws, Wa=Wa, bp=bp)
    for name, k in (("Wq", Da), ("Wk", Da), ("Wv", Dv)):
        A[name] = np.array([s[f"ch.{name}.{i}"] for i in range(k)], np.float64)
    wc = np.array([s[f"ch.Wc.{c}"] for c in range(C)], np.float64)
    A.update(Wc=wc[:, 1:], bc=wc[:, 0])
    bw = np.zeros(V)
    for c in range(C):
        for t, v in zip(classes[c], s[f"ch.kb.{c}"]):
            bw[t] = v
    A["bw"] = bw
    skip = ("ch.e.", "ch.Wp.", "ch.Wq.", "ch.Wk.", "ch.Wv.", "ch.Wc.", "ch.k.", "ch.kb.", "ch.t.", "ch.v.", "ch.c.")
    meta = [(k, v) for k, v in s.items() if not k.startswith(skip) and k != "ch.ready"]
    texts_of = [t for i in range((V + 999) // 1000) for t in s[f"ch.t.{i}"]]
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for k, mp in s.items():
        if k.startswith("ch.v."):
            vocab.update(mp)
        elif k.startswith("ch.c."):
            vocab.update({"##" + p: i for p, i in mp.items()})
    return A, meta, classes, texts_of, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of an exported model with number lists")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="chat13m")
    a = ap.parse_args()
    A, meta, classes, texts_of, vocab = from_export(a.src)
    Q = pack(A)
    tmp = a.out.rstrip("/\\") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    count, stored = write(tmp, Q, meta, classes, vocab, texts_of, a.name)
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.replace(tmp, a.out)
    print(f"written {a.out}: {count} files, {stored / 1e6:.1f} MB; K {Q['K']:.2f}, embedding scale {Q['se']:.5f}")


if __name__ == "__main__":
    main()
