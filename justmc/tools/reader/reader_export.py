#!/usr/bin/env python3
"""
Export of the Millennium reader (reader_train.py) for JustMC: one manifest that the library import
(~ml model import, ~ai install) walks through.

  reader-000.json ..   the BiLSTM as a normal library model (config, LayerNorm, gate rows) and
  save-reader.json     {"save": "reader"}: kept as the saved model "reader"
  rd-*.json            {"store": {...}}: save variables of the demo
                         rd.v.<b> / rd.c.<b>  vocabulary maps (word starts / continuations without "##"),
                                              b = (code of 1st char + 37 * code of 2nd char) mod 251
                         rd.e.<c>, rd.es.<c>  8-bit embeddings: ROWS rows of 128 characters U+0100+code
                                              per text, scale of every row (e = scale * (code - 128))
                         rd.P.<j>, rd.PA, rd.pb   projection rows, -384 * sum(P_j), bias
                         rd.Ws.<k>, rd.We.<k>     [b | W] rows of the start / end heads
                         rd.ns, rd.ne             "no answer" rows over [1 | q]
                         rd.stop, rd.uk_only, rd.ru_only, sizes; rd.ready = 1 last
Every file has at most 24000 bytes of UTF-8 (JustMC cuts web responses at 25000 characters; counting
bytes keeps the files whole even if the cut were in bytes). The 8-bit codes are 2-byte characters.

    python reader_export.py --model out/reader.pt --vocab data/vocab.json --out ../../weights/reader --tau 0.3
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_text as R  # noqa: E402

MAX_BYTES = 24000
ROWS = 80             # embedding rows per text
CP0 = 0x100           # code 0 -> U+0100 (Latin Extended: printable, 2 bytes in UTF-8)
NB = 251              # vocabulary maps per kind
GATES = "fico"


def fmt(v, digits=10):
    return float(f"%.{digits}g" % float(v))


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def size(text):
    return len(text.encode("utf-8"))


def bucket(key):
    c1 = ord(key[1]) if len(key) > 1 else 0
    return (ord(key[0]) + 37 * c1) % NB


class Packer:
    """writes the files of the manifest; store entries are packed into files of <= MAX_BYTES"""

    def __init__(self, out):
        self.out = out
        self.manifest = []
        self.cur, self.cur_name = {}, None
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(out):
            if f.endswith(".json"):
                os.remove(os.path.join(out, f))

    def put(self, name, text):
        assert size(text) <= MAX_BYTES, (name, size(text))
        open(os.path.join(self.out, name), "w", encoding="utf-8").write(text)
        self.manifest.append(name)

    def store(self, prefix, items):
        """items: [(save variable, value)] -> files prefix-000.json, ..."""
        files, cur = [], {}
        for k, v in items:
            if cur and size(dumps({"store": {**cur, k: v}})) > MAX_BYTES:
                files.append(cur)
                cur = {}
            cur[k] = v
            if size(dumps({"store": cur})) > MAX_BYTES:
                raise SystemExit(f"{k} alone is longer than {MAX_BYTES} bytes")
        if cur:
            files.append(cur)
        for i, f in enumerate(files):
            self.put(f"{prefix}-{i:03d}.json", dumps({"store": f}))

    def finish(self, name="reader"):
        return write_manifest(self.out, name, self.manifest)


def write_manifest(out, name, files):
    """<name>.json: the list of files. JustMC cuts a response at 25000 characters, so a list that is longer
    becomes a manifest of manifests <name>-list-<i>.json (the importer queues their files in order)."""
    text = dumps({"format": "millennium", "manifest": files})
    if size(text) > MAX_BYTES:
        parts, cur = [], []
        for f in files:
            if cur and size(dumps({"format": "millennium", "manifest": cur + [f]})) > MAX_BYTES:
                parts.append(cur)
                cur = []
            cur.append(f)
        parts.append(cur)
        names = [f"{name}-list-{i}.json" for i in range(len(parts))]
        for pn, p in zip(names, parts):
            open(os.path.join(out, pn), "w").write(dumps({"format": "millennium", "manifest": p}))
        text = dumps({"format": "millennium", "manifest": names})
    open(os.path.join(out, name + ".json"), "w").write(text)
    return len(files)


def library_parts(P, pd, hid):
    """the BiLSTM + question attention as a Millennium model (input mode GRID, attention pooling)"""
    nin = 2 * pd + 4
    cfg = {"inputSize": nin, "hiddenSize": hid, "numLayers": 1, "bidirectional": 1, "inputMode": 3,
           "poolingMode": 3, "learningRate": 0.001, "dropoutRate": 0.0, "recurrentDropoutRate": 0.0,
           "weightDecay": 0.0, "gradientClip": 2.0, "labelSmoothing": 0.0, "batchSize": 16,
           "step": 0, "optT": 0, "decisionThreshold": 0.5}
    head = {"format": "millennium", "version": 6, "config": cfg,
            "embWy": [0.0] * nin, "embWp": [0.0] * nin, "embB": [0.0] * nin,
            "attnW": [fmt(v) for v in P["aW"]], "attnB": fmt(P["aB"]),
            "headV": [0.0] * (2 * hid), "headBias": 0.0}
    ln = {"format": "millennium", "ln": {str(d): {"g": [fmt(v) for v in P["gam"][d]],
                                                   "b": [fmt(v) for v in P["bet"][d]]} for d in (0, 1)}}
    rows = []
    for d in (0, 1):
        for n in range(hid):
            for gi, g in enumerate(GATES):
                row = [P["b"][d, gi, n]] + list(P["Wx"][d, gi, n]) + list(P["Wh"][d, gi, n])
                rows.append([d, g, n, [fmt(v) for v in row]])
    files = [dumps(head), dumps(ln)]
    cur = []
    for r in rows:
        if cur and size(dumps({"format": "millennium", "rows": cur + [r]})) > MAX_BYTES:
            files.append(dumps({"format": "millennium", "rows": cur}))
            cur = []
        cur.append(r)
    files.append(dumps({"format": "millennium", "rows": cur}))
    return files


def vocab_items(vocab):
    buckets = {}
    for tok, i in vocab.items():
        if tok in ("[UNK]", "[PAD]"):
            continue
        if tok.startswith(R.CONT):
            key, kind = tok[len(R.CONT):], "c"
        else:
            key, kind = tok, "v"
        buckets.setdefault(f"rd.{kind}.{bucket(key)}", {})[key] = i
    return sorted(buckets.items(), key=lambda kv: (kv[0].split(".")[1], int(kv[0].split(".")[2])))


def quantize(emb):
    m = emb.abs().max(1).values.clamp_min(1e-8)
    scale = m / 127.0
    codes = torch.clamp(torch.round(emb / scale[:, None]) + 128, 0, 255).to(torch.int64)
    return codes, scale


def embedding_items(emb):
    codes, scale = quantize(emb)
    items = []
    for c in range(0, emb.shape[0], ROWS):
        text = "".join(chr(CP0 + int(x)) for x in codes[c:c + ROWS].reshape(-1))
        items.append((f"rd.e.{c // ROWS}", text))
        items.append((f"rd.es.{c // ROWS}", [fmt(s) for s in scale[c:c + ROWS]]))
    return items


def net_items(P, pd):
    items = []
    Pm = [[fmt(v, 12) for v in row] for row in P["P"]]
    for j, row in enumerate(Pm):
        items.append((f"rd.P.{j}", row))
    off = 128 + CP0
    items.append(("rd.PA", [-off * sum(row) for row in Pm]))
    items.append(("rd.pb", [fmt(v) for v in P["pb"]]))
    out = P["Ws"].shape[0]
    for k in range(out):
        items.append((f"rd.Ws.{k}", [fmt(P["bs"][k])] + [fmt(v) for v in P["Ws"][k]]))
        items.append((f"rd.We.{k}", [fmt(P["be"][k])] + [fmt(v) for v in P["We"][k]]))
    items.append(("rd.ns", [fmt(v) for v in P["ns"]]))
    items.append(("rd.ne", [fmt(v) for v in P["ne"]]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.3)
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu")
    st = ck["state"]
    P = {k: v.detach().double().numpy() for k, v in st.items() if k != "emb"}
    hid, pd = ck["hidden"], ck["pd"]
    vocab = json.load(open(a.vocab, encoding="utf-8"))
    n_par = sum(v.numel() for v in st.values())
    pk = Packer(a.out)
    for i, t in enumerate(library_parts(P, pd, hid)):
        pk.put(f"reader-{i:03d}.json", t)
    pk.put("save-reader.json", dumps({"format": "millennium", "save": "reader"}))
    stop = {w: 1 for w in sorted(R.ALL_STOP)}
    meta = [("rd.stop", stop), ("rd.uk_only", {w: 1 for w in sorted(R.UK_ONLY)}),
            ("rd.ru_only", {w: 1 for w in sorted(R.RU_ONLY)}), ("rd.pd", pd), ("rd.hid", hid),
            ("rd.nin", 2 * pd + 4), ("rd.rows", ROWS), ("rd.vocab", len(vocab)), ("rd.params", n_par),
            ("rd.tau0", a.tau)]
    pk.store("rd-meta", meta)
    pk.store("rd-net", net_items(P, pd))
    pk.store("rd-vocab", vocab_items(vocab))
    pk.store("rd-emb", embedding_items(st["emb"].detach().float()))
    pk.put("rd-ready.json", dumps({"store": {"rd.ready": 1}}))
    n = pk.finish()
    print(f"written {a.out}/reader.json (manifest) + {n} files; {n_par} parameters, hidden {hid}")


if __name__ == "__main__":
    main()
