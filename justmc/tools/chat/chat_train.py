#!/usr/bin/env python3
"""
The Millennium chat model: a word-piece language model that writes the reply token by token.

    x_t = E[token]                          embedding, 96 numbers (8-bit in JustMC, tied with the output)
    h_t = LSTM(x_t, h_t-1)                  the Millennium cell with LayerNorm (the library LSTM in JustMC)
    o_t = Wp h_t + bp                       96 numbers
    P(next) = P(class | o) * P(token | class, o)
        class logits  Wc o + bc             ~200 classes of tokens grouped by frequency
        token logits  E[token] . o + bw     only the tokens of that class (at most 128)
So one generated token costs one LSTM step and ~300 short dot products instead of 20 000.
Tokens that never occur in replies have no class and are never generated.

    python chat_train.py --data data --out out --hidden 128 --epochs 3
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

CAP = 128            # tokens per class at most


def make_classes(X, M, V, n_classes=160):
    """frequency binning of the predicted tokens: frequent tokens in small classes"""
    tgt = X[:, 1:][M[:, :-1] == 1]
    freq = np.bincount(tgt, minlength=V).astype(np.float64)
    order = [t for t in np.argsort(-freq, kind="stable") if freq[t] > 0]
    target = freq.sum() / n_classes
    classes, cur, mass = [], [], 0.0
    for t in order:
        cur.append(int(t))
        mass += freq[t]
        if mass >= target or len(cur) >= CAP:
            classes.append(cur)
            cur, mass = [], 0.0
    if cur:
        classes.append(cur)
    return classes


class ChatLM(torch.nn.Module):
    def __init__(self, V, classes, dim=96, hid=128):
        super().__init__()
        self.V, self.D, self.H = V, dim, hid
        self.emb = torch.nn.Parameter(torch.randn(V, dim) * 0.1)
        s = math.sqrt(6.0 / (dim + hid))
        self.Wx = torch.nn.Parameter(torch.empty(4, hid, dim).uniform_(-s, s))
        self.Wh = torch.nn.Parameter(torch.empty(4, hid, hid).uniform_(-1 / math.sqrt(hid), 1 / math.sqrt(hid)))
        b = torch.zeros(4, hid)
        b[0] = 1.0
        self.b = torch.nn.Parameter(b)
        self.gam = torch.nn.Parameter(torch.ones(hid))
        self.bet = torch.nn.Parameter(torch.zeros(hid))
        self.Wp = torch.nn.Parameter(torch.randn(dim, hid) / math.sqrt(hid))
        self.bp = torch.nn.Parameter(torch.zeros(dim))
        C = len(classes)
        self.Wc = torch.nn.Parameter(torch.randn(C, dim) * 0.05)
        self.bc = torch.nn.Parameter(torch.zeros(C))
        self.bw = torch.nn.Parameter(torch.zeros(V))
        cls_of = torch.full((V,), -1, dtype=torch.long)
        pos_in = torch.zeros(V, dtype=torch.long)
        for c, toks in enumerate(classes):
            for k, t in enumerate(toks):
                cls_of[t] = c
                pos_in[t] = k
        self.register_buffer("cls_of", cls_of)
        self.register_buffer("pos_in", pos_in)
        self.members = [torch.tensor(t, dtype=torch.long) for t in classes]

    def run(self, x, h=None, c=None, emb=None):
        """x: (B, T) token ids -> o (B, T, D), last state"""
        E = self.emb if emb is None else emb
        B, T = x.shape
        H = self.H
        xe = E[x] @ self.Wx.reshape(4 * H, -1).t() + self.b.reshape(4 * H)
        Wh = self.Wh.reshape(4 * H, H).t()
        h = x.new_zeros(B, H, dtype=torch.float32) if h is None else h
        c = x.new_zeros(B, H, dtype=torch.float32) if c is None else c
        hs = []
        for t in range(T):
            z = xe[:, t] + h @ Wh
            f = torch.sigmoid(z[:, :H].clamp(-50, 50))
            i = torch.sigmoid(z[:, H:2 * H].clamp(-50, 50))
            g = torch.tanh(z[:, 2 * H:3 * H].clamp(-50, 50))
            o = torch.sigmoid(z[:, 3 * H:].clamp(-50, 50))
            c = f * c + i * g
            pre = o * torch.tanh(c)
            mu = pre.mean(-1, keepdim=True)
            var = ((pre - mu) ** 2).mean(-1, keepdim=True)
            h = self.gam * (pre - mu) / torch.sqrt(var + 1e-5) + self.bet
            hs.append(h)
        hs = torch.stack(hs, 1)
        return hs @ self.Wp.t() + self.bp, h, c

    def loss(self, x, m, emb=None):
        """mean -log P of the tokens marked in m (position t predicts x[t + 1])"""
        o, _, _ = self.run(x[:, :-1], emb=emb)
        sel = m[:, :-1].bool()
        o = o[sel]
        y = x[:, 1:][sel]
        cy = self.cls_of[y]
        lc = torch.nn.functional.cross_entropy(o @ self.Wc.t() + self.bc, cy, reduction="sum")
        E = self.emb if emb is None else emb
        lw = o.new_zeros(())
        for c in torch.unique(cy).tolist():
            idx = (cy == c).nonzero(as_tuple=True)[0]
            mem = self.members[c]
            logits = o[idx] @ E[mem].t() + self.bw[mem]
            lw = lw + torch.nn.functional.cross_entropy(logits, self.pos_in[y[idx]], reduction="sum")
        return (lc + lw) / max(1, len(y)), len(y)


def quantize(emb):
    m = emb.abs().max(1).values.clamp_min(1e-8)
    scale = m / 127.0
    codes = torch.clamp(torch.round(emb / scale[:, None]) + 128, 0, 255)
    return codes, scale


def batches(idx, lengths, bs, rng):
    """batches of similar length (less padding)"""
    idx = idx[np.argsort(lengths[idx] + rng.random_sample(len(idx)) * 4)]
    out = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    rng.shuffle(out)
    return out


def evaluate(m, X, M, L, idx, emb=None, bs=512):
    m.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            T = int(L[b].max())
            l, k = m.loss(torch.as_tensor(X[b, :T], dtype=torch.long), torch.as_tensor(M[b, :T]), emb)
            tot += float(l.detach()) * k
            n += k
    m.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--qat", type=int, default=1, help="last epochs with 8-bit embeddings in the forward pass")
    ap.add_argument("--init", help="checkpoint to fine-tune (its token classes are kept)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    D = np.load(os.path.join(a.data, "chat_data.npz"))
    meta = json.load(open(os.path.join(a.data, "chat_meta.json")))
    X, M = D["X"], D["M"]
    V = meta["n_vocab"] + len(meta["special"])
    L = (X != 1).sum(1)
    ck = torch.load(a.init, map_location="cpu") if a.init else None
    classes = ck["classes"] if ck else make_classes(X, M, V)
    m = ChatLM(V, classes, a.dim, a.hidden)
    if ck:
        m.load_state_dict(ck["state"])
    n_par = sum(p.numel() for p in m.parameters())
    mac = 4 * a.hidden * (a.dim + a.hidden + 1) + a.dim * (a.hidden + 1) + len(classes) * (a.dim + 1) + CAP * (a.dim + 1)
    print(f"parameters {n_par}; {len(classes)} classes ({sum(len(c) for c in classes)} tokens can be generated); "
          f"<= {mac / 1000:.0f} k multiply-adds per generated token", flush=True)
    tr = np.where(D["split"] == 0)[0]
    va = np.where(D["split"] == 1)[0]
    if ck:
        # fine-tuning keeps the token classes: sequences with a reply token outside them are left out
        cls = m.cls_of.numpy()
        bad = ((M[:, :-1] == 1) & (cls[X[:, 1:]] < 0)).any(1)
        print(f"fine-tuning: {int(bad.sum())} sequences with tokens outside the classes left out")
        tr, va = tr[~bad[tr]], va[~bad[va]]
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-6)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.03)
    for ep in range(a.epochs):
        qat = ep >= a.epochs - a.qat
        t0 = time.time()
        tot, n = 0.0, 0
        for bi, b in enumerate(batches(tr, L, a.batch, rng)):
            T = int(L[b].max())
            emb = None
            if qat:
                codes, scale = quantize(m.emb.detach())
                q = scale[:, None] * (codes - 128)
                emb = m.emb + (q - m.emb).detach()          # straight-through 8-bit embeddings
            l, k = m.loss(torch.as_tensor(X[b, :T], dtype=torch.long), torch.as_tensor(M[b, :T]), emb)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(l.detach()) * k
            n += k
            if bi % 200 == 0:
                print(f"  epoch {ep + 1} batch {bi}: loss {tot / max(1, n):.3f} ({time.time() - t0:.0f} s)", flush=True)
        codes, scale = quantize(m.emb.detach())
        qemb = scale[:, None] * (codes - 128)
        vl, vq = evaluate(m, X, M, L, va), evaluate(m, X, M, L, va, qemb)
        print(f"epoch {ep + 1}/{a.epochs} train {tot / n:.3f} | val {vl:.3f} (ppl {math.exp(vl):.1f}), 8-bit {vq:.3f} "
              f"(ppl {math.exp(vq):.1f}) ({time.time() - t0:.0f} s)", flush=True)
        os.makedirs(a.out, exist_ok=True)
        torch.save({"state": m.state_dict(), "classes": classes, "hidden": a.hidden, "dim": a.dim, "V": V,
                    "special": meta["special"]}, os.path.join(a.out, "chat.pt"))


if __name__ == "__main__":
    main()
