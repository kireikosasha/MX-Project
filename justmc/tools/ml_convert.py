#!/usr/bin/env python3
"""
ml_convert.py — convert Millennium 5 (RNNModelML, format RNN6 v6) weights between the MX
binary ``.dat`` file and the JSON format understood by the JustMC port (``ml_import_json``).

Examples
  # MX plugin model -> one JSON file (for `~mx model import <url>`)
  python ml_convert.py dat2json ../../src/main/resources/ml/m1-rnn.dat quark.json

  # the same, split into small parts (config+heads, then one LSTM layer-direction per file)
  python ml_convert.py dat2json m1-rnn.dat quark.json --split

  # JustMC export (or merged parts) -> MX plugin .dat (drop it into plugins/MX/models/)
  python ml_convert.py json2dat model.json m1-rnn.dat part1.json part2.json ...

  # check / describe a file
  python ml_convert.py info m1-rnn.dat

The binary layout mirrors RNNModelML.saveToFile():
  int MAGIC(0x524E4E36) int VERSION(6)
  int inputSize int hiddenSize int numLayers bool bidirectional
  int inputMode int poolingMode
  double lr dr rdr wd gc ls
  int batchSize long step long optT
  arr embWy arr embWp arr embB
  int numLayers bool bidirectional  { for l: layer(fwd) [layer(bwd)] }
      layer = Wf Wi Wc Wo Uf Ui Uc Uo bf bi bc bo lnGamma lnBeta   (arr = int n + n doubles)
  arr attnW double attnB arr headV double headBias double decisionThreshold
"""
import json
import struct
import sys

MAGIC = 0x524E4E36
VERSION = 6
LAYER_KEYS = ["Wf", "Wi", "Wc", "Wo", "Uf", "Ui", "Uc", "Uo", "bf", "bi", "bc", "bo", "lnGamma", "lnBeta"]
CONFIG_KEYS = ["inputSize", "hiddenSize", "numLayers", "bidirectional", "inputMode", "poolingMode",
               "learningRate", "dropoutRate", "recurrentDropoutRate", "weightDecay", "gradientClip",
               "labelSmoothing", "batchSize", "step", "optT", "decisionThreshold"]


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
        if n < 0:
            return None
        return [self.f64() for _ in range(n)]

    def left(self):
        return len(self.b) - self.p


def read_dat(path):
    r = Reader(open(path, "rb").read())
    if r.i32() != MAGIC or r.i32() != VERSION:
        raise SystemExit("bad model version (expected RNN6 / 6)")
    cfg = {}
    cfg["inputSize"] = r.i32()
    cfg["hiddenSize"] = r.i32()
    cfg["numLayers"] = r.i32()
    cfg["bidirectional"] = 1 if r.boolean() else 0
    cfg["inputMode"] = r.i32()
    cfg["poolingMode"] = r.i32()
    for k in ["learningRate", "dropoutRate", "recurrentDropoutRate", "weightDecay", "gradientClip", "labelSmoothing"]:
        cfg[k] = r.f64()
    cfg["batchSize"] = r.i32()
    cfg["step"] = r.i64()
    cfg["optT"] = r.i64()
    m = {"format": "millennium", "version": VERSION, "config": cfg}
    m["embWy"] = r.arr()
    m["embWp"] = r.arr()
    m["embB"] = r.arr()
    layers = r.i32()
    bi = r.boolean()
    lay = {}
    for l in range(layers):
        for d in ([0, 1] if bi else [0]):
            lay[str(l * 2 + d)] = {k: r.arr() for k in LAYER_KEYS}
    m["layers"] = lay
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
    w(">i", int(cfg["inputSize"]))
    w(">i", int(cfg["hiddenSize"]))
    w(">i", int(cfg["numLayers"]))
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
            layer = m["layers"][str(l * 2 + d)]
            for k in LAYER_KEYS:
                arr(layer[k])
    arr(m["attnW"])
    w(">d", float(scalar(m["attnB"])))
    arr(m["headV"])
    w(">d", float(scalar(m["headBias"])))
    w(">d", float(cfg["decisionThreshold"]))
    open(path, "wb").write(bytes(out))


def scalar(v):
    return v[0] if isinstance(v, list) else v


def merge(parts):
    m = {}
    for p in parts:
        for k, v in p.items():
            if k == "layers":
                m.setdefault("layers", {}).update(v)
            elif k == "config":
                m.setdefault("config", {}).update(v)
            else:
                m[k] = v
    return m


def dump(obj, path, digits):
    def rnd(x):
        if isinstance(x, list):
            return [rnd(y) for y in x]
        if isinstance(x, dict):
            return {k: rnd(y) for k, y in x.items()}
        if isinstance(x, float) and digits:
            return float(f"{x:.{digits}g}")
        return x
    open(path, "w").write(json.dumps(rnd(obj), separators=(",", ":")))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[1]
    digits = None
    if "--digits" in argv:
        digits = int(argv[argv.index("--digits") + 1])
    if cmd == "dat2json":
        m = read_dat(argv[2])
        out = argv[3]
        if "--split" in argv:
            base = out[:-5] if out.endswith(".json") else out
            head = {k: v for k, v in m.items() if k != "layers"}
            dump(head, f"{base}-part0.json", digits)
            names = [f"{base}-part0.json"]
            for i, (d, layer) in enumerate(sorted(m["layers"].items(), key=lambda kv: int(kv[0]))):
                p = {"format": "millennium", "version": VERSION, "layers": {d: layer}}
                dump(p, f"{base}-part{i + 1}.json", digits)
                names.append(f"{base}-part{i + 1}.json")
            print("written:", " ".join(names))
        else:
            dump(m, out, digits)
            print("written:", out)
    elif cmd == "json2dat":
        parts = [json.load(open(p)) for p in [argv[2]] + argv[4:] if not p.startswith("--")]
        write_dat(merge(parts), argv[3])
        print("written:", argv[3])
    elif cmd == "info":
        src = argv[2]
        m = read_dat(src) if src.endswith(".dat") else merge([json.load(open(p)) for p in argv[2:]])
        cfg = m["config"]
        n = 3 * cfg["inputSize"] + len(m["attnW"]) + 1 + len(m["headV"]) + 1
        for layer in m["layers"].values():
            n += sum(len(layer[k]) for k in LAYER_KEYS)
        print(json.dumps(cfg, indent=1))
        print("parameters:", n)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
