#!/usr/bin/env python3
"""
The Millennium reader: finds the answer to a question inside a short text (or says there is none).
The forward pass is the one demo_reader.jc runs in JustMC:

  e(token)      embedding row, stored as 8-bit codes with a scale per row (looked up, no math)
  p = P e + pb  projection to PD numbers
  al_t          question words attended by context word t: softmax_i(p_t . p_qi), sum_i a_ti p_qi
  x_t           [p_t | al_t | features_t]  (question tokens: [p | 0 | 0])
  BiLSTM        Millennium LSTM cell with LayerNorm (ml_lstm_forward), shared by question and text
  q             attention pooling of the question states (ml_pool_attention)
  start / end   h_t . (Ws q + bs), h_t . (We q + be); "no answer" logits ns = ws . [1|q], ne = we . [1|q]

    python reader_train.py --data data/reader_data.npz --out out --hidden 24 --epochs 12
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_text as R  # noqa: E402

GATES = "fico"
NFEAT = 4


class Reader(torch.nn.Module):
    def __init__(self, vocab, dim=128, pd=32, hid=24):
        super().__init__()
        self.V, self.D, self.PD, self.H = vocab, dim, pd, hid
        self.emb = torch.nn.Parameter(torch.randn(vocab, dim) * 0.1)
        self.P = torch.nn.Parameter(torch.randn(pd, dim) * (1.0 / math.sqrt(dim)))
        self.pb = torch.nn.Parameter(torch.zeros(pd))
        nin = 2 * pd + NFEAT
        self.nin = nin
        s_in = math.sqrt(6.0 / (nin + hid))
        self.Wx = torch.nn.Parameter(torch.empty(2, 4, hid, nin).uniform_(-s_in, s_in))
        self.Wh = torch.nn.Parameter(torch.empty(2, 4, hid, hid).uniform_(-1 / math.sqrt(hid), 1 / math.sqrt(hid)))
        b = torch.zeros(2, 4, hid)
        b[:, 0] = 1.0
        self.b = torch.nn.Parameter(b)
        self.gam = torch.nn.Parameter(torch.ones(2, hid))
        self.bet = torch.nn.Parameter(torch.zeros(2, hid))
        out = 2 * hid
        self.aW = torch.nn.Parameter(torch.empty(out).uniform_(-0.1, 0.1))
        self.aB = torch.nn.Parameter(torch.zeros(()))
        self.Ws = torch.nn.Parameter(torch.randn(out, out) * (1 / math.sqrt(out)))
        self.bs = torch.nn.Parameter(torch.zeros(out))
        self.We = torch.nn.Parameter(torch.randn(out, out) * (1 / math.sqrt(out)))
        self.be = torch.nn.Parameter(torch.zeros(out))
        self.ns = torch.nn.Parameter(torch.zeros(out + 1))
        self.ne = torch.nn.Parameter(torch.zeros(out + 1))

    # ---- Millennium BiLSTM over padded sequences (lengths n); the reverse direction starts at n-1
    def lstm(self, x, n):
        B, T, _ = x.shape
        H = self.H
        t_idx = torch.arange(T)[None, :].expand(B, T)
        rev = torch.where(t_idx < n[:, None], n[:, None] - 1 - t_idx, t_idx)
        xr = torch.gather(x, 1, rev[:, :, None].expand(B, T, x.shape[2]))
        outs = []
        for d, xs in ((0, x), (1, xr)):
            Wx = self.Wx[d].reshape(4 * H, -1)
            Wh = self.Wh[d].reshape(4 * H, H)
            bb = self.b[d].reshape(4 * H)
            xp = xs @ Wx.t() + bb
            h = x.new_zeros(B, H)
            c = x.new_zeros(B, H)
            hs = []
            for t in range(T):
                z = xp[:, t] + h @ Wh.t()
                f = torch.sigmoid(z[:, :H].clamp(-50, 50))
                i = torch.sigmoid(z[:, H:2 * H].clamp(-50, 50))
                g = torch.tanh(z[:, 2 * H:3 * H].clamp(-50, 50))
                o = torch.sigmoid(z[:, 3 * H:].clamp(-50, 50))
                c = f * c + i * g
                pre = o * torch.tanh(c)
                mu = pre.mean(-1, keepdim=True)
                var = ((pre - mu) ** 2).mean(-1, keepdim=True)
                h = self.gam[d] * (pre - mu) / torch.sqrt(var + 1e-5) + self.bet[d]
                hs.append(h)
            hs = torch.stack(hs, 1)
            if d == 1:
                hs = torch.gather(hs, 1, rev[:, :, None].expand(B, T, H))
            outs.append(hs)
        return torch.cat(outs, -1)

    def embed(self, ids, emb=None):
        e = (self.emb if emb is None else emb)[ids]
        return e @ self.P.t() + self.pb

    def forward(self, cid, cn, feats, qid, qn, emb=None, drop=0.0):
        B, T = cid.shape
        Q = qid.shape[1]
        pc = self.embed(cid, emb)
        pq = self.embed(qid, emb)
        if drop > 0:
            pc = torch.nn.functional.dropout(pc, drop, self.training)
            pq = torch.nn.functional.dropout(pq, drop, self.training)
        qmask = torch.arange(Q)[None, :] < qn[:, None]
        sc = torch.einsum("btd,bqd->btq", pc, pq).masked_fill(~qmask[:, None, :], -1e30)
        a = torch.softmax(sc, -1)
        al = torch.einsum("btq,bqd->btd", a, pq)
        xc = torch.cat([pc, al, feats], -1)
        xq = torch.cat([pq, torch.zeros_like(pq), pq.new_zeros(B, Q, NFEAT)], -1)
        hc = self.lstm(xc, cn)
        hq = self.lstm(xq, qn)
        s = hq @ self.aW + self.aB
        s = s.masked_fill(~qmask, -1e30)
        e = torch.exp(s - s.max(-1, keepdim=True).values) * qmask
        w = e / (e.sum(-1, keepdim=True) + 1e-10)
        q = torch.einsum("bq,bqd->bd", w, hq)
        us = q @ self.Ws.t() + self.bs
        ue = q @ self.We.t() + self.be
        st = torch.einsum("btd,bd->bt", hc, us)
        en = torch.einsum("btd,bd->bt", hc, ue)
        q1 = torch.cat([q.new_ones(B, 1), q], -1)
        return st, en, q1 @ self.ns, q1 @ self.ne


def quantize(emb):
    """8-bit codes per row: e = scale * (code - 128), code 0..255"""
    m = emb.abs().max(1).values.clamp_min(1e-8)
    scale = m / 127.0
    codes = torch.clamp(torch.round(emb / scale[:, None]) + 128, 0, 255)
    return codes, scale


def batches(idx, bs, rng):
    idx = idx.copy()
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def tensors(D, b):
    return (torch.as_tensor(D["ids"][b], dtype=torch.long), torch.as_tensor(D["n"][b], dtype=torch.long),
            torch.as_tensor(D["f"][b]), torch.as_tensor(D["q"][b], dtype=torch.long),
            torch.as_tensor(np.maximum(D["qn"][b], 1), dtype=torch.long),
            torch.as_tensor(D["ws"][b], dtype=torch.bool), torch.as_tensor(D["we"][b], dtype=torch.bool),
            torch.as_tensor(D["s"][b], dtype=torch.long), torch.as_tensor(D["e"][b], dtype=torch.long))


def logits_with_null(st, en, ns, ne, cn, ws, we):
    T = st.shape[1]
    valid = torch.arange(T)[None, :] < cn[:, None]
    st = st.masked_fill(~(valid & ws), -1e30)
    en = en.masked_fill(~(valid & we), -1e30)
    return torch.cat([ns[:, None], st], 1), torch.cat([ne[:, None], en], 1)


def decode(ls, le, maxlen=12):
    """probabilities -> (p_null, best start, best end, best span probability)"""
    ps = torch.softmax(ls, 1)
    pe = torch.softmax(le, 1)
    pn = ps[:, 0] * pe[:, 0]
    sp = ps[:, 1:, None] * pe[:, None, 1:]
    T = sp.shape[1]
    i = torch.arange(T)
    ok = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < maxlen)
    sp = sp * ok
    flat = sp.reshape(sp.shape[0], -1)
    best, arg = flat.max(1)
    return pn, arg // T, arg % T, best


def evaluate(m, D, idx, emb=None, bs=1024):
    m.eval()
    res = {k: [] for k in ("pn", "bs", "be", "best", "s", "e", "lang", "kind")}
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            cid, cn, f, qid, qn, ws, we, s, e = tensors(D, b)
            st, en, ns, ne = m(cid, cn, f, qid, qn, emb)
            ls, le = logits_with_null(st, en, ns, ne, cn, ws, we)
            pn, bsv, bev, best = decode(ls, le)
            for k, v in (("pn", pn), ("bs", bsv), ("be", bev), ("best", best), ("s", s), ("e", e)):
                res[k].append(v.numpy())
            res["lang"].append(D["lang"][b])
            res["kind"].append(D["kind"][b])
    m.train()
    return {k: np.concatenate(v) for k, v in res.items()}


def report(r, tag):
    has = r["s"] >= 0
    lines = []
    for name, sel in (("all", np.ones_like(has)), ("en", r["lang"] == 0), ("ru", r["lang"] == 1), ("uk", r["lang"] == 2)):
        h = has & sel
        tok_f1 = []
        for s, e, bs, be in zip(r["s"][h], r["e"][h], r["bs"][h], r["be"][h]):
            inter = max(0, min(e, be) - max(s, bs) + 1)
            if inter == 0:
                tok_f1.append(0.0)
            else:
                p = inter / (be - bs + 1)
                rr = inter / (e - s + 1)
                tok_f1.append(2 * p * rr / (p + rr))
        f1 = float(np.mean(tok_f1)) if tok_f1 else 0.0
        em = float(((r["bs"] == r["s"]) & (r["be"] == r["e"]))[h].mean()) if h.any() else 0.0
        lines.append(f"{name}: span F1 {f1 * 100:.1f} EM {em * 100:.1f}")
    # paranoid operating point: answer only if best > p_null and best >= tau; precision = answered with F1 > 0.5
    out = []
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5):
        ans = (r["best"] > r["pn"]) & (r["best"] >= tau)
        good = ans & has & (np.maximum(0, np.minimum(r["e"], r["be"]) - np.maximum(r["s"], r["bs"]) + 1) * 2 >
                            np.maximum(r["e"] - r["s"] + 1, r["be"] - r["bs"] + 1))
        prec = good.sum() / max(1, ans.sum())
        cov = good.sum() / max(1, has.sum())
        out.append(f"tau {tau}: answers {ans.mean() * 100:.0f}%, right {prec * 100:.1f}%, finds {cov * 100:.1f}% of answerable")
    print(f"[{tag}] " + " | ".join(lines))
    print(f"[{tag}] " + " | ".join(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--pd", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    D = dict(np.load(a.data))
    vocab = json.load(open(os.path.join(os.path.dirname(a.data), "vocab.json")))
    tr = np.where(D["split"] == 0)[0]
    va = np.where(D["split"] == 1)[0]
    va = va[rng.permutation(len(va))[:20000]]
    m = Reader(len(vocab), a.dim, a.pd, a.hidden)
    n_par = sum(p.numel() for p in m.parameters())
    mac = (R.MAXC + R.MAXQ) * a.pd * a.dim + R.MAXC * R.MAXQ * a.pd * 2 + (R.MAXC + R.MAXQ) * 2 * 4 * a.hidden * (1 + m.nin + a.hidden)
    print(f"parameters {n_par} ({len(vocab)} x {a.dim} embeddings = {len(vocab) * a.dim}); ~{mac / 1e6:.2f} M multiply-adds per question")
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    best, best_state = -1, None
    for ep in range(a.epochs):
        t0 = time.time()
        tot, cnt = 0.0, 0
        for b in batches(tr, a.batch, rng):
            cid, cn, f, qid, qn, ws, we, s, e = tensors(D, b)
            st, en, ns, ne = m(cid, cn, f, qid, qn, drop=0.1)
            ls, le = logits_with_null(st, en, ns, ne, cn, ws, we)
            loss = torch.nn.functional.cross_entropy(ls, s + 1) + torch.nn.functional.cross_entropy(le, e + 1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
            cnt += len(b)
        r = evaluate(m, D, va)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / cnt:.4f} ({time.time() - t0:.0f} s)")
        report(r, "val")
        has = r["s"] >= 0
        score = float(((r["bs"] == r["s"]) & (r["be"] == r["e"]))[has].mean())
        if score > best:
            best = score
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    codes, scale = quantize(m.emb.detach())
    qemb = scale[:, None] * (codes - 128)
    print("with 8-bit embeddings:")
    report(evaluate(m, D, va, emb=qemb), "val q8")
    te = np.where(D["split"] == 2)[0]
    report(evaluate(m, D, te, emb=qemb), "xquad-ru q8")
    os.makedirs(a.out, exist_ok=True)
    torch.save({"state": m.state_dict(), "hidden": a.hidden, "pd": a.pd, "dim": a.dim, "vocab": len(vocab)},
               os.path.join(a.out, "reader.pt"))


if __name__ == "__main__":
    main()
