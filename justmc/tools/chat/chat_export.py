#!/usr/bin/env python3
"""
Export of the Millennium chat model (chat_train.py) for JustMC, one manifest for the library import:

  chat-000.json ..   the LSTM as a library model (input 96, one direction, LayerNorm), saved as "chat"
  ch-*.json          {"store": {...}}:
                       ch.e.<c>, ch.es.<c>   8-bit embeddings (vocabulary order, ROWS rows of 96 characters
                                             U+0100+code per text) and the scale of every row
                       ch.Wp.<j>             [bp | Wp] rows: o = Wp h + bp
                       ch.Wc.<c>             [bc | Wc] rows: class logits
                       ch.k.<c>, ch.kb.<c>   tokens of class c and their output biases
                       ch.t.<n>              token texts, 1000 per list (to write the reply)
                       ch.U / B / E / W      special tokens; ch.cW / ch.pW: class and place of [W] in it ...
                       settings, ch.ready = 1 last
  ai.json            reader + chat in one manifest (for a world without the reader)
Every file has at most 24000 bytes.

    python chat_export.py --model out/chat.pt --vocab ../reader/data/vocab.json --out ../../weights/chat
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reader"))
from reader_export import Packer, dumps, fmt, size, MAX_BYTES, CP0  # noqa: E402

ROWS = 112
BASE = "https://raw.githubusercontent.com/kireikosasha/MX-Project/master/justmc/weights"
GATES = "fico"


def library_parts(P, dim, hid):
    cfg = {"inputSize": dim, "hiddenSize": hid, "numLayers": 1, "bidirectional": 0, "inputMode": 3,
           "poolingMode": 0, "learningRate": 0.001, "dropoutRate": 0.0, "recurrentDropoutRate": 0.0,
           "weightDecay": 0.0, "gradientClip": 1.0, "labelSmoothing": 0.0, "batchSize": 16,
           "step": 0, "optT": 0, "decisionThreshold": 0.5}
    head = {"format": "millennium", "version": 6, "config": cfg,
            "embWy": [0.0] * dim, "embWp": [0.0] * dim, "embB": [0.0] * dim,
            "attnW": [0.0] * hid, "attnB": 0.0, "headV": [0.0] * hid, "headBias": 0.0}
    ln = {"format": "millennium", "ln": {"0": {"g": [fmt(v) for v in P["gam"]], "b": [fmt(v) for v in P["bet"]]}}}
    files = [dumps(head), dumps(ln)]
    cur = []
    for n in range(hid):
        for gi, g in enumerate(GATES):
            row = [P["b"][gi, n]] + list(P["Wx"][gi, n]) + list(P["Wh"][gi, n])
            r = [0, g, n, [fmt(v, 7) for v in row]]
            if cur and size(dumps({"format": "millennium", "rows": cur + [r]})) > MAX_BYTES:
                files.append(dumps({"format": "millennium", "rows": cur}))
                cur = []
            cur.append(r)
    files.append(dumps({"format": "millennium", "rows": cur}))
    return files


def quantize(emb):
    m = emb.abs().max(1).values.clamp_min(1e-8)
    scale = m / 127.0
    codes = torch.clamp(torch.round(emb / scale[:, None]) + 128, 0, 255).to(torch.int64)
    return codes, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topc", type=int, default=4)
    ap.add_argument("--topw", type=int, default=8)
    ap.add_argument("--wiki", type=float, default=0.05, help="P([W]) from which the reader answers")
    ap.add_argument("--sure", type=float, default=0.3, help="P([W]) from which a failed search ends in 'I don't know' "
                                                            "(below it the chat model replies after all)")
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu")
    st, classes, sp = ck["state"], ck["classes"], ck["special"]
    P = {k: v.detach().double().numpy() for k, v in st.items() if k not in ("emb", "cls_of", "pos_in")}
    hid, dim, V = ck["hidden"], ck["dim"], ck["V"]
    vocab = json.load(open(a.vocab, encoding="utf-8"))
    texts = [""] * V
    for t, i in vocab.items():
        texts[i] = t
    for name, i in sp.items():
        texts[i] = name
    n_par = sum(v.numel() for k, v in st.items() if k not in ("cls_of", "pos_in"))
    pk = Packer(a.out)
    for i, t in enumerate(library_parts(P, dim, hid)):
        pk.put(f"chat-{i:03d}.json", t)
    pk.put("save-chat.json", dumps({"format": "millennium", "save": "chat"}))
    where = {t: (c, k) for c, toks in enumerate(classes) for k, t in enumerate(toks)}
    meta = [("ch.hid", hid), ("ch.dim", dim), ("ch.V", V), ("ch.C", len(classes)), ("ch.rows", ROWS),
            ("ch.params", n_par), ("ch.temp0", a.temp), ("ch.topc", a.topc), ("ch.topw", a.topw),
            ("ch.wiki0", a.wiki), ("ch.sure0", a.sure), ("ch.max", 40)]
    for name in ("U", "B", "E", "W"):
        i = sp[f"[{name}]"]
        c, k = where.get(i, (-1, -1))
        meta += [(f"ch.{name}", i), (f"ch.c{name}", c), (f"ch.p{name}", k)]
    pk.store("ch-meta", meta)
    net = []
    for j in range(dim):
        net.append((f"ch.Wp.{j}", [fmt(P["bp"][j])] + [fmt(v, 8) for v in P["Wp"][j]]))
    for c in range(len(classes)):
        net.append((f"ch.Wc.{c}", [fmt(P["bc"][c])] + [fmt(v, 8) for v in P["Wc"][c]]))
    for c, toks in enumerate(classes):
        net.append((f"ch.k.{c}", toks))
        net.append((f"ch.kb.{c}", [fmt(P["bw"][t], 8) for t in toks]))
    pk.store("ch-net", net)
    pk.store("ch-text", [(f"ch.t.{n}", texts[n * 1000:(n + 1) * 1000]) for n in range((V + 999) // 1000)])
    codes, scale = quantize(st["emb"].detach().float())
    emb = []
    for c in range(0, V, ROWS):
        emb.append((f"ch.e.{c // ROWS}", "".join(chr(CP0 + int(x)) for x in codes[c:c + ROWS].reshape(-1))))
        emb.append((f"ch.es.{c // ROWS}", [fmt(s) for s in scale[c:c + ROWS]]))
    pk.store("ch-emb", emb)
    pk.put("ch-ready.json", dumps({"store": {"ch.ready": 1}}))
    n = pk.finish("chat")
    open(os.path.join(a.out, "ai.json"), "w").write(dumps({"format": "millennium", "manifest": [
        f"{BASE}/reader/reader.json", f"{BASE}/chat/chat.json"]}))
    print(f"written {a.out}/chat.json (manifest) + {n} files, ai.json (reader + chat); {n_par} parameters")


if __name__ == "__main__":
    main()
