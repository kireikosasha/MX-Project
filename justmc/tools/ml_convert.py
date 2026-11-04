#!/usr/bin/env python3
"""
ml_convert.py - convert Millennium RNN weights (RNNModelML, format RNN6 v6) between the MX
binary .dat file and the JSON files of the JustMC port.

JustMC cuts web responses at 25000 characters, so a model is written as many small JSON files:
  <name>.json        manifest {"format": "millennium", "manifest": ["<name>-000.json", ...]}
  <name>-000.json    config, embedding, attention, head
  <name>-001.json    LayerNorm of every layer        {"ln": {"<d>": {"g": [...], "b": [...]}}}
  <name>-002.json..  gate rows                       {"rows": [[d, "f|i|c|o", n, [b, W..., U...]], ...]}
In JustMC:  ~ml model import <url of <name>.json>   (the other files must be next to it)

Usage
  python ml_convert.py dat2json m1-rnn.dat quark-e.json            # MX -> JustMC files + manifest
  python ml_convert.py json2dat m1-rnn.dat part1.json part2.json ... # JustMC export / files -> MX .dat
  python ml_convert.py info m1-rnn.dat | info file.json ...
Options:  --max-chars N (default 24000)   --single (one JSON file, only for small models)
"""
import glob
import json
import os
import struct
import sys

MAGIC = 0x524E4E36
VERSION = 6
GATES = ["f", "i", "c", "o"]
LAYER_KEYS = ["Wf", "Wi", "Wc", "Wo", "Uf", "Ui", "Uc", "Uo", "bf", "bi", "bc", "bo", "lnGamma", "lnBeta"]


class Reader:
    def __init__(self, b):
        self.b = b
        self.p = 0

    def take(self, fmt, n):
        v = struct.unpack(fmt, self.b[self.p:self.p + n])[0]
        self.p += n
        return v

    def i32(self):
        return self.take(">i", 4)

    def i64(self):
        return self.take(">q", 8)

    def f64(self):
        return self.take(">d", 8)

    def boolean(self):
        v = self.b[self.p] != 0
        self.p += 1
        return v

    def arr(self):
        n = self.i32()
        return None if n < 0 else [self.f64() for _ in range(n)]

    def left(self):
        return len(self.b) - self.p


def read_dat(path):
    r = Reader(open(path, "rb").read())
    if r.i32() != MAGIC or r.i32() != VERSION:
        raise SystemExit("bad model version (expected RNN6 / 6)")
    cfg = {"inputSize": r.i32(), "hiddenSize": r.i32(), "numLayers": r.i32(),
           "bidirectional": 1 if r.boolean() else 0, "inputMode": r.i32(), "poolingMode": r.i32()}
    for k in ["learningRate", "dropoutRate", "recurrentDropoutRate", "weightDecay", "gradientClip", "labelSmoothing"]:
        cfg[k] = r.f64()
    cfg["batchSize"] = r.i32()
    cfg["step"] = r.i64()
    cfg["optT"] = r.i64()
    m = {"format": "millennium", "version": VERSION, "config": cfg,
         "embWy": r.arr(), "embWp": r.arr(), "embB": r.arr()}
    layers = r.i32()
    bi = r.boolean()
    m["layers"] = {}
    for l in range(layers):
        for d in ([0, 1] if bi else [0]):
            m["layers"][str(l * 2 + d)] = {k: r.arr() for k in LAYER_KEYS}
    m["attnW"] = r.arr()
    m["attnB"] = r.f64()
    m["headV"] = r.arr()
    m["headBias"] = r.f64()
    cfg["decisionThreshold"] = r.f64() if r.left() >= 8 else 0.60
    return m


def write_dat(m, path):
    cfg = m["config"]
    out = bytearray()
    w = lambda fmt, v: out.extend(struct.pack(fmt, v))

    def arr(a):
        w(">i", len(a))
        for x in a:
            w(">d", float(x))

    w(">i", MAGIC)
    w(">i", VERSION)
    for k in ["inputSize", "hiddenSize", "numLayers"]:
        w(">i", int(cfg[k]))
    out.append(1 if int(cfg["bidirectional"]) else 0)
    w(">i", int(cfg["inputMode"]))
    w(">i", int(cfg["poolingMode"]))
    for k in ["learningRate", "dropoutRate", "recurrentDropoutRate", "weightDecay", "gradientClip", "labelSmoothing"]:
        w(">d", float(cfg[k]))
    w(">i", int(cfg["batchSize"]))
    w(">q", int(cfg["step"]))
    w(">q", int(cfg["optT"]))
    arr(m["embWy"])
    arr(m["embWp"])
    arr(m["embB"])
    w(">i", int(cfg["numLayers"]))
    bi = int(cfg["bidirectional"])
    out.append(1 if bi else 0)
    for l in range(int(cfg["numLayers"])):
        for d in ([0, 1] if bi else [0]):
            for k in LAYER_KEYS:
                arr(m["layers"][str(l * 2 + d)][k])
    arr(m["attnW"])
    w(">d", float(scalar(m["attnB"])))
    arr(m["headV"])
    w(">d", float(scalar(m["headBias"])))
    w(">d", float(cfg["decisionThreshold"]))
    open(path, "wb").write(bytes(out))


def scalar(v):
    return v[0] if isinstance(v, list) else v


def dims(cfg, d):
    h = int(cfg["hiddenSize"])
    n_in = int(cfg["inputSize"]) if int(d) // 2 == 0 else (2 * h if int(cfg["bidirectional"]) else h)
    return n_in, h


def to_rows(m):
    """layers (flat Java arrays) -> [[d, gate, n, [b, W..., U...]], ...]"""
    rows = []
    for d, lay in sorted(m["layers"].items(), key=lambda kv: int(kv[0])):
        n_in, h = dims(m["config"], d)
        for n in range(h):
            for g in GATES:
                W, U, b = lay["W" + g], lay["U" + g], lay["b" + g]
                rows.append([int(d), g, n, [b[n]] + W[n * n_in:(n + 1) * n_in] + U[n * h:(n + 1) * h]])
    return rows


def merge(parts):
    """files (full model, header, ln, rows, JustMC export chunks) -> one model dict"""
    m, rows, ln = {}, [], {}
    for p in parts:
        for k, v in p.items():
            if k == "layers":
                m.setdefault("layers", {}).update(v)
            elif k == "config":
                m.setdefault("config", {}).update(v)
            elif k == "rows":
                rows.extend(v)
            elif k == "ln":
                ln.update(v)
            elif k != "manifest":
                m[k] = v
    if rows or ln:
        cfg = m["config"]
        lay = m.setdefault("layers", {})
        for d, g, n, row in rows:
            n_in, h = dims(cfg, d)
            L = lay.setdefault(str(int(d)), {})
            for key, size in (("W" + g, h * n_in), ("U" + g, h * h), ("b" + g, h)):
                L.setdefault(key, [0.0] * size)
            n = int(n)
            L["b" + g][n] = row[0]
            L["W" + g][n * n_in:(n + 1) * n_in] = row[1:1 + n_in]
            L["U" + g][n * h:(n + 1) * h] = row[1 + n_in:]
        for d, v in ln.items():
            L = lay.setdefault(str(int(float(d))), {})
            L["lnGamma"], L["lnBeta"] = v["g"], v["b"]
    return m


def text(obj):
    return json.dumps(obj, separators=(",", ":"))


def write_chunks(m, out, max_chars):
    base = out[:-5] if out.endswith(".json") else out
    stem = os.path.basename(base)
    files = [{k: v for k, v in m.items() if k != "layers"}]
    files.append({"format": "millennium", "ln": {d: {"g": L["lnGamma"], "b": L["lnBeta"]}
                                                  for d, L in m["layers"].items()}})
    cur = []
    for r in to_rows(m):
        if cur and len(text({"format": "millennium", "rows": cur + [r]})) > max_chars:
            files.append({"format": "millennium", "rows": cur})
            cur = []
        cur.append(r)
    if cur:
        files.append({"format": "millennium", "rows": cur})
    names = []
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for i, f in enumerate(files):
        t = text(f)
        if len(t) > max_chars:
            raise SystemExit(f"file {i} has {len(t)} characters (> {max_chars})")
        name = f"{stem}-{i:03d}.json"
        open(os.path.join(os.path.dirname(out) or ".", name), "w").write(t)
        names.append(name)
    open(out, "w").write(text({"format": "millennium", "manifest": names}))
    print(f"written: {out} (manifest) + {len(names)} files of <= {max_chars} characters")


def load_parts(paths):
    parts = []
    for p in paths:
        for f in sorted(glob.glob(p)) or [p]:
            parts.append(json.load(open(f)))
    return parts


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    opts, args, max_chars = set(), [], 24000
    it = iter(argv[1:])
    for a in it:
        if a == "--max-chars":
            max_chars = int(next(it))
        elif a.startswith("--"):
            opts.add(a)
        else:
            args.append(a)
    cmd = args[0]
    if cmd == "dat2json":
        m = read_dat(args[1])
        if "--single" in opts:
            open(args[2], "w").write(text(m))
            print("written:", args[2])
        else:
            write_chunks(m, args[2], max_chars)
    elif cmd == "json2dat":
        write_dat(merge(load_parts(args[2:])), args[1])
        print("written:", args[1])
    elif cmd == "info":
        m = read_dat(args[1]) if args[1].endswith(".dat") else merge(load_parts(args[1:]))
        n = 3 * m["config"]["inputSize"] + len(m["attnW"]) + 1 + len(m["headV"]) + 1
        n += sum(len(L[k]) for L in m["layers"].values() for k in LAYER_KEYS)
        print(json.dumps(m["config"], indent=1))
        print("parameters:", n)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
