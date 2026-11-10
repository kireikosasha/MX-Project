#!/usr/bin/env python3
"""
Export of the Millennium chat model v2 (chat_train.py) for JustMC, one manifest for the library import:

  chat-000.json ..   the LSTM as a library model (input = embedding, one direction, LayerNorm), saved as "chat"
  ch-*.json          {"store": {...}}:
      ch.e.<c>              8-bit embeddings: ROWS rows of D characters U+0100 + code per text; value = s0 * (cp - 384).
                            The scale is the same for the whole table, so it is folded into the weights that read an
                            embedding: JustMC feeds the code points themselves to the LSTM and to the output rows.
      ch.Wp.<j>             output rows [bp | Wp (h) | Ws (code points) | Wa (ctx)]
      ch.Wq.<i>, ch.Wk.<i>, ch.Wv.<i>   attention: query, key, value rows over an LSTM state
      ch.Wc.<c>             [bc | Wc] rows: class logits
      ch.k.<c>, ch.kb.<c>   tokens of class c and their output biases
      ch.t.<n>              token texts, 1000 per list (to write the reply)
      ch.v.<b>, ch.c.<b>    vocabulary maps of the tokenizer (word starts / continuations), as in the reader
      ch.h0, ch.c0          the LSTM state after [U] from zero (a reply starts from it: one step less)
      settings, special tokens; ch.ready = 1 last
  ai.json            reader + chat in one manifest (for a world without the reader)
Every file has at most 24000 bytes.

    python chat_export.py --model out/chat.pt --vocab vocab.json --out ../../weights/chat
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reader"))
import reader_text as R  # noqa: E402
from reader_export import Packer, dumps, fmt, size, MAX_BYTES, CP0, bucket  # noqa: E402
sys.path.insert(0, HERE)
from chat_ref import lstm_step  # noqa: E402

BASE = "https://raw.githubusercontent.com/kireikosasha/MX-Project/master/justmc/weights"
GATES = "fico"
OFF = CP0 + 128               # code point of the value 0
DIGITS = 4                    # significant digits of the matrices


def rnd(v):
    return [fmt(x, DIGITS) for x in v]


def fold(bias, wx, s0):
    """weights over an embedding -> weights over its code points: [bias', wx * s0] with bias' = bias - OFF * sum(wx * s0)"""
    w = rnd(wx * s0)
    return fmt(float(bias) - OFF * sum(w), 9), w


def library_parts(P, D, H, s0):
    """-> JSON texts of the library model, its rows {(gate, n): row} and LayerNorm (for the start state)"""
    nin = D
    cfg = {"inputSize": nin, "hiddenSize": H, "numLayers": 1, "bidirectional": 0, "inputMode": 3,
           "poolingMode": 0, "learningRate": 0.001, "dropoutRate": 0.0, "recurrentDropoutRate": 0.0,
           "weightDecay": 0.0, "gradientClip": 1.0, "labelSmoothing": 0.0, "batchSize": 16,
           "step": 0, "optT": 0, "decisionThreshold": 0.5}
    head = {"format": "millennium", "version": 6, "config": cfg,
            "embWy": [0.0] * nin, "embWp": [0.0] * nin, "embB": [0.0] * nin,
            "attnW": [0.0] * H, "attnB": 0.0, "headV": [0.0] * H, "headBias": 0.0}
    lnv = {"g": [fmt(v) for v in P["gam"]], "b": [fmt(v) for v in P["bet"]]}
    ln = {"format": "millennium", "ln": {"0": lnv}}
    files = [dumps(head), dumps(ln)]
    cur, rows = [], {}
    for n in range(H):
        for gi, g in enumerate(GATES):
            k = gi * H + n
            b, wx = fold(P["b"][k], P["Wx"][k], s0)
            r = [0, g, n, [b] + wx + rnd(P["Wh"][k])]
            rows[(g, n)] = r[3]
            if cur and size(dumps({"format": "millennium", "rows": cur + [r]})) > MAX_BYTES:
                files.append(dumps({"format": "millennium", "rows": cur}))
                cur = []
            cur.append(r)
    files.append(dumps({"format": "millennium", "rows": cur}))
    return files, rows, lnv


def codes_of(emb, s0):
    return (torch.clamp(torch.round(emb / s0), -127, 127) + 128).to(torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--typ", type=float, default=0.9, help="mass of the typical set (typical sampling)")
    ap.add_argument("--ks", type=float, default=0.6, help="Kolmogorov-Smirnov distance from which a step is predictable")
    ap.add_argument("--topc", type=int, default=4)
    ap.add_argument("--topw", type=int, default=8)
    ap.add_argument("--wiki", type=float, default=0.1, help="P([W]) from which the reader answers")
    ap.add_argument("--sure", type=float, default=0.4, help="P([W]) from which a failed search ends in 'I don't know' "
                                                            "(below it the chat model replies after all)")
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    st, classes, sp, arch, s0 = ck["state"], ck["classes"], ck["special"], ck["arch"], float(ck["scale"])
    P = {k: v.detach().double().numpy() for k, v in st.items() if k not in ("emb", "cls_of", "pos_in")}
    H, D, Da, Dv, V = arch["hid"], arch["dim"], arch["da"], arch["dv"], ck["V"]
    rows = (MAX_BYTES - 400) // (2 * D)
    vocab = json.load(open(a.vocab, encoding="utf-8"))
    texts = [""] * V
    for t, i in vocab.items():
        texts[i] = t
    for name, i in sp.items():
        texts[i] = name
    n_par = sum(v.numel() for k, v in st.items() if k not in ("cls_of", "pos_in"))
    pk = Packer(a.out)
    lib, rows_lstm, lnv = library_parts(P, D, H, s0)
    for i, t in enumerate(lib):
        pk.put(f"chat-{i:03d}.json", t)
    codes = codes_of(st["emb"].detach().float(), s0)
    cps_u = [float(CP0 + int(x)) for x in codes[sp["[U]"]]]
    h0, c0 = lstm_step(rows_lstm, lnv, H, cps_u, [0.0] * H, [0.0] * H)
    pk.put("save-chat.json", dumps({"format": "millennium", "save": "chat"}))
    where = {t: (c, k) for c, toks in enumerate(classes) for k, t in enumerate(toks)}
    meta = [("ch.ver", 2), ("ch.hid", H), ("ch.dim", D), ("ch.dv", Dv), ("ch.da", Da), ("ch.nin", D), ("ch.V", V),
            ("ch.C", len(classes)), ("ch.rows", rows), ("ch.params", n_par), ("ch.s0", fmt(s0, 9)), ("ch.so", fmt(-OFF * s0, 9)),
            ("ch.temp0", a.temp), ("ch.samp0", 1), ("ch.typ0", a.typ), ("ch.ks0", a.ks), ("ch.topc", a.topc), ("ch.topw", a.topw), ("ch.wiki0", a.wiki), ("ch.sure0", a.sure),
            ("ch.max", 36), ("ch.maxm", 24)]
    for name in ("U", "B", "E", "W"):
        i = sp[f"[{name}]"]
        c, k = where.get(i, (-1, -1))
        meta += [(f"ch.{name}", i), (f"ch.c{name}", c), (f"ch.p{name}", k)]
    meta += [("ch.h0", [fmt(v, 12) for v in h0]), ("ch.c0", [fmt(v, 12) for v in c0])]
    pk.store("ch-meta", meta)
    net = []
    for j in range(D):
        b, ws = fold(P["bp"][j], P["Ws"][j], s0)
        net.append((f"ch.Wp.{j}", [b] + rnd(P["Wp"][j]) + ws + rnd(P["Wa"][j])))
    for name in ("Wq", "Wk", "Wv"):
        for i in range(P[name].shape[0]):
            net.append((f"ch.{name}.{i}", rnd(P[name][i])))
    for c in range(len(classes)):
        net.append((f"ch.Wc.{c}", [fmt(P["bc"][c], 9)] + rnd(P["Wc"][c])))
    for c, toks in enumerate(classes):
        net.append((f"ch.k.{c}", toks))
        net.append((f"ch.kb.{c}", [fmt(P["bw"][t], 6) for t in toks]))
    pk.store("ch-net", net)
    pk.store("ch-text", [(f"ch.t.{n}", texts[n * 1000:(n + 1) * 1000]) for n in range((V + 999) // 1000)])
    buckets = {}
    for tok, i in vocab.items():
        if tok in ("[UNK]", "[PAD]"):
            continue
        key, kind = (tok[len(R.CONT):], "c") if tok.startswith(R.CONT) else (tok, "v")
        buckets.setdefault(f"ch.{kind}.{bucket(key)}", {})[key] = i
    pk.store("ch-vocab", sorted(buckets.items()))
    pk.store("ch-emb", [(f"ch.e.{c // rows}", "".join(chr(CP0 + int(x)) for x in codes[c:c + rows].reshape(-1)))
                        for c in range(0, V, rows)])
    pk.put("ch-ready.json", dumps({"store": {"ch.ready": 1}}))
    n = pk.finish("chat")
    open(os.path.join(a.out, "ai.json"), "w").write(dumps({"format": "millennium", "manifest": [
        f"{BASE}/reader/reader.json", f"{BASE}/chat/chat.json"]}))
    print(f"written {a.out}/chat.json (manifest) + {n} files, ai.json (reader + chat); {n_par} parameters")


if __name__ == "__main__":
    main()
