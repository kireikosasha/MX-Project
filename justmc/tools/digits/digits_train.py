#!/usr/bin/env python3
"""
Trains the ten digit models of the Millennium digit demo and writes them in the JustMC format.

Every digit d gets its own Millennium RNN (one-vs-rest): input mode GRID (one row of the N x N
picture per time step), one bidirectional LSTM layer with LayerNorm, attention pooling and the
sigmoid head - the same forward pass as ml_net.jc, so the weights run unchanged in JustMC.
The ten models are trained together (batched) with softmax cross-entropy over their ten logits
plus the binary loss of every model.

    python digits_train.py --data data.npz --out weights/digits --hidden 24 --epochs 30
"""
import argparse
import json
import math
import os

import numpy as np
import torch

GATES = "fico"   # row order of the library: forget, input, candidate (c), output


class DigitModels(torch.nn.Module):
    """K independent Millennium models (1 layer, bidirectional) evaluated in one pass."""

    def __init__(self, k, n_in, hid):
        super().__init__()
        self.k, self.n_in, self.hid = k, n_in, hid
        s_in = math.sqrt(6.0 / (n_in + hid))
        s_h = math.sqrt(1.0 / hid)
        self.Wx = torch.nn.Parameter(torch.empty(k, 2, 4, hid, n_in).uniform_(-s_in, s_in))
        self.Wh = torch.nn.Parameter(torch.empty(k, 2, 4, hid, hid).uniform_(-s_h, s_h))
        b = torch.zeros(k, 2, 4, hid)
        b[:, :, 0] = 1.0  # forget gate bias
        self.b = torch.nn.Parameter(b)
        self.gam = torch.nn.Parameter(torch.ones(k, 2, hid))
        self.bet = torch.nn.Parameter(torch.zeros(k, 2, hid))
        out = 2 * hid
        self.aW = torch.nn.Parameter(torch.empty(k, out).uniform_(-0.1, 0.1))
        self.aB = torch.nn.Parameter(torch.zeros(k))
        self.hV = torch.nn.Parameter(torch.empty(k, out).uniform_(-0.1, 0.1))
        self.hB = torch.nn.Parameter(torch.zeros(k))

    def forward(self, x, dropout=0.0):
        """x: [B, T, n_in] -> logits [K, B]   (models and directions are batched as KD = K*2)"""
        k, hid = self.k, self.hid
        B, T, n_in = x.shape
        kd = k * 2
        Wx = self.Wx.reshape(kd, 4 * hid, n_in)                       # rows: gate-major (f, i, c, o)
        Wh = self.Wh.reshape(kd, 4 * hid, hid)
        b = self.b.reshape(kd, 1, 4 * hid)
        # input projection of every step for every model/direction: [KD, T, B, 4H]
        xt = x.transpose(0, 1).reshape(T * B, n_in)
        xp = torch.matmul(xt, Wx.transpose(1, 2)).reshape(kd, T, B, 4 * hid) + b[:, None]
        rev = torch.arange(kd) % 2 == 1
        h = x.new_zeros(kd, B, hid)
        c = x.new_zeros(kd, B, hid)
        gam = self.gam.reshape(kd, 1, hid)
        bet = self.bet.reshape(kd, 1, hid)
        WhT = Wh.transpose(1, 2)
        outs = []
        for s in range(T):
            # direction 0 reads t = s, direction 1 reads t = T-1-s
            zx = torch.where(rev[:, None, None], xp[:, T - 1 - s], xp[:, s])
            z = zx + torch.bmm(h, WhT)
            f = torch.sigmoid(z[..., :hid].clamp(-50, 50))
            i = torch.sigmoid(z[..., hid:2 * hid].clamp(-50, 50))
            g = torch.tanh(z[..., 2 * hid:3 * hid].clamp(-50, 50))
            o = torch.sigmoid(z[..., 3 * hid:].clamp(-50, 50))
            c = f * c + i * g
            pre = o * torch.tanh(c)
            mu = pre.mean(-1, keepdim=True)
            var = ((pre - mu) ** 2).mean(-1, keepdim=True)
            h = gam * (pre - mu) / torch.sqrt(var + 1e-5) + bet
            outs.append(h)
        Hs = torch.stack(outs, 2).reshape(k, 2, B, T, hid)             # step order
        fwd = Hs[:, 0]
        bwd = Hs[:, 1].flip(2)                                          # back to time order
        H = torch.cat([fwd, bwd], -1)                                   # [K, B, T, 2H]
        if dropout > 0:
            H = torch.nn.functional.dropout(H, dropout, self.training)
        sc = torch.matmul(H, self.aW[:, None, :, None]).squeeze(-1) + self.aB[:, None, None]
        e = torch.exp(sc - sc.max(-1, keepdim=True).values)
        a = e / (e.sum(-1, keepdim=True) + 1e-10)
        pooled = torch.matmul(a.unsqueeze(2), H).squeeze(2)            # [K, B, 2H]
        return torch.matmul(pooled, self.hV[:, :, None]).squeeze(-1) + self.hB[:, None]


def to_rows(x, n):
    """[B, n*n] 0/1 -> [B, n, n] float rows"""
    return torch.as_tensor(x.reshape(len(x), n, n), dtype=torch.float32)


def accuracy(model, x, y, n, bs=4096):
    model.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            pred.append(model(to_rows(x[i:i + bs], n)).argmax(0).numpy())
    model.train()
    pred = np.concatenate(pred)
    return float((pred == y).mean()), pred


# ----------------------------------------------------------------------------- export
def fmt(v):
    return float("%.10g" % float(v))


def model_files(m, k, n_in, prefix, max_chars=24000):
    """Millennium JSON files (header, LayerNorm, gate rows) of model k -> list of (name, text)"""
    hid = m.hid
    P = {name: p.detach().double().numpy() for name, p in m.named_parameters()}
    cfg = {"inputSize": n_in, "hiddenSize": hid, "numLayers": 1, "bidirectional": 1, "inputMode": 3,
           "poolingMode": 3, "learningRate": 0.001, "dropoutRate": 0.1, "recurrentDropoutRate": 0.0,
           "weightDecay": 0.0005, "gradientClip": 2.0, "labelSmoothing": 0.0, "batchSize": 16,
           "step": 0, "optT": 0, "decisionThreshold": 0.5}
    head = {"format": "millennium", "version": 6, "config": cfg,
            "embWy": [0.0] * n_in, "embWp": [0.0] * n_in, "embB": [0.0] * n_in,
            "attnW": [fmt(v) for v in P["aW"][k]], "attnB": fmt(P["aB"][k]),
            "headV": [fmt(v) for v in P["hV"][k]], "headBias": fmt(P["hB"][k])}
    ln = {"format": "millennium", "ln": {str(d): {"g": [fmt(v) for v in P["gam"][k, d]],
                                                   "b": [fmt(v) for v in P["bet"][k, d]]} for d in (0, 1)}}
    rows = []
    for d in (0, 1):
        for n in range(hid):
            for gi, g in enumerate(GATES):
                row = [P["b"][k, d, gi, n]] + list(P["Wx"][k, d, gi, n]) + list(P["Wh"][k, d, gi, n])
                rows.append([d, g, n, [fmt(v) for v in row]])
    texts = [json.dumps(head, separators=(",", ":")), json.dumps(ln, separators=(",", ":"))]
    cur = []
    for r in rows:
        if cur and len(json.dumps({"format": "millennium", "rows": cur + [r]}, separators=(",", ":"))) > max_chars:
            texts.append(json.dumps({"format": "millennium", "rows": cur}, separators=(",", ":")))
            cur = []
        cur.append(r)
    texts.append(json.dumps({"format": "millennium", "rows": cur}, separators=(",", ":")))
    return [(f"{prefix}{k}-{i:03d}.json", t) for i, t in enumerate(texts)]


def export(m, n_in, out_dir, name="digits"):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for k in range(m.k):
        for fname, text in model_files(m, k, n_in, "digit"):
            assert len(text) <= 24000, (fname, len(text))
            open(os.path.join(out_dir, fname), "w").write(text)
            manifest.append(fname)
        sname = f"save-digit{k}.json"
        open(os.path.join(out_dir, sname), "w").write(json.dumps({"format": "millennium", "save": f"digit{k}"}))
        manifest.append(sname)
    open(os.path.join(out_dir, name + ".json"), "w").write(
        json.dumps({"format": "millennium", "manifest": manifest}, separators=(",", ":")))
    print(f"written {out_dir}/{name}.json (manifest) + {len(manifest)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    D = np.load(a.data)
    xtr, ytr = D["xtr"].reshape(len(D["xtr"]), -1), D["ytr"].astype(np.int64)
    xva, yva, sva = D["xva"].reshape(len(D["xva"]), -1), D["yva"].astype(np.int64), D["sva"]
    n = int(round(math.sqrt(xtr.shape[1])))
    m = DigitModels(10, n, a.hidden)
    print("parameters per model:", sum(p.numel() for p in m.parameters()) // 10 + 3 * n)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(xtr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    eye = torch.eye(10)
    best, best_state = 0.0, None
    for ep in range(a.epochs):
        perm = np.random.permutation(len(xtr))
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb = to_rows(xtr[idx], n)
            yb = torch.as_tensor(ytr[idx])
            logits = m(xb, dropout=0.1)                       # [10, B]
            ce = torch.nn.functional.cross_entropy(logits.t(), yb, label_smoothing=0.05)
            target = eye[yb].t()                               # [10, B] one-vs-rest labels
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=torch.full((10, 1), 3.0))
            loss = ce + 0.5 * bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        acc, pred = accuracy(m, xva, yva, n)
        per_src = [float((pred[sva == s] == yva[sva == s]).mean()) for s in range(3)]
        print(f"epoch {ep + 1:2d}/{a.epochs}  loss {tot / len(perm):.4f}  validation {acc * 100:.2f}%  "
              f"(fonts {per_src[0] * 100:.1f}%, handwritten {per_src[1] * 100:.1f}%, pixel {per_src[2] * 100:.1f}%)", flush=True)
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    acc, pred = accuracy(m, xva, yva, n)
    conf = np.zeros((10, 10), int)
    for t, p in zip(yva, pred):
        conf[t, p] += 1
    print(f"best validation accuracy {acc * 100:.2f}%")
    print("confusion (rows = true digit):")
    for d in range(10):
        print(d, " ".join(f"{v:5d}" for v in conf[d]))
    torch.save(m.state_dict(), os.path.join(os.path.dirname(a.data), "digits_model.pt"))
    export(m, n, a.out)


if __name__ == "__main__":
    main()
