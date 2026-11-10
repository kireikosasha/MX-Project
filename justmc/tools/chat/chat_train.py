#!/usr/bin/env python3
"""
The Millennium chat model (v2): a word-piece language model that writes the reply token by token.

    x_t   = E[token]                               embedding (8-bit in JustMC), tied with the output
    h_t   = LSTM(x_t, h_t-1)                       the Millennium cell with LayerNorm (library step in JustMC)
    ctx_t = sum_j a_j (Wv h_j),  a = softmax_j((Wq h_t) . (Wk h_j))   attention over the message token states
    o_t   = Wp [1 | h_t | x_t | ctx_t]             output vector (the x_t part is a bigram shortcut)
    P(next) = P(class | o) * P(token | class, o)
        class logits  Wc o + bc                    classes of tokens by frequency (frequent tokens alone)
        token logits  E[token] . o + bw            only the tokens of that class
JustMC pays per action (about 50 per LSTM neuron and step), not per multiplication, so the rows are long
(wide embedding, shortcut) and the number of rows is what is kept small.

    python chat_train.py --data data --out out --hidden 256 --dim 256 --epochs 10 [--device cuda]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

CAP = 160            # tokens per class at most
MAX_M = 24           # message tokens = attention slots
CLIP = 5.0           # 8-bit embeddings: one scale for the table, values clipped at CLIP standard deviations


def make_classes(X, M, V, alpha=2.0, beta=6.0):
    """Frequency-ordered classes that minimise the JustMC actions per generated token:
    alpha * (number of classes) + beta * (expected size of the class of the next token)."""
    tgt = X[:, 1:][M[:, :-1] == 1]
    freq = np.bincount(tgt, minlength=V).astype(np.float64)
    order = [int(t) for t in np.argsort(-freq, kind="stable") if freq[t] > 0]
    p = freq[order] / freq[order].sum()
    P = np.concatenate([[0.0], np.cumsum(p)])
    n = len(order)
    best = np.full(n + 1, np.inf)
    back = np.zeros(n + 1, np.int64)
    best[0] = 0.0
    for i in range(1, n + 1):
        j = np.arange(max(0, i - CAP), i)
        cost = best[j] + alpha + beta * (P[i] - P[j]) * (i - j)
        k = int(np.argmin(cost))
        best[i], back[i] = cost[k], j[k]
    cuts, i = [], n
    while i > 0:
        cuts.append((int(back[i]), i))
        i = int(back[i])
    return [order[a:b] for a, b in reversed(cuts)]


def roles(X, sp):
    """per position: message token (between [U] and [B]), its attention slot, [U]"""
    is_u = X == sp["[U]"]
    is_b = X == sp["[B]"]
    depth = np.cumsum(is_u, 1) - np.cumsum(is_b, 1)
    is_msg = (depth == 1) & ~is_u & (X != 1)
    pos = np.arange(X.shape[1])[None, :]
    last_u = np.maximum.accumulate(np.where(is_u, pos, -1), 1)
    slot = np.clip(pos - last_u - 1, 0, MAX_M - 1)
    return is_u, is_msg, slot.astype(np.int64)


class ChatLM(torch.nn.Module):
    def __init__(self, V, classes, dim=256, hid=256, da=64, dv=128):
        super().__init__()
        self.V, self.D, self.H, self.Da, self.Dv = V, dim, hid, da, dv
        H, D = hid, dim
        self.emb = torch.nn.Parameter(torch.randn(V, D) * 0.1)
        s = math.sqrt(6.0 / (D + H))
        self.Wx = torch.nn.Parameter(torch.empty(4 * H, D).uniform_(-s, s))
        self.Wh = torch.nn.Parameter(torch.empty(4 * H, H).uniform_(-1 / math.sqrt(H), 1 / math.sqrt(H)))
        b = torch.zeros(4, H)
        b[0] = 1.0
        self.b = torch.nn.Parameter(b.reshape(4 * H))
        self.gam = torch.nn.Parameter(torch.ones(H))
        self.bet = torch.nn.Parameter(torch.zeros(H))
        self.Wp = torch.nn.Parameter(torch.randn(D, H) / math.sqrt(H))
        self.bp = torch.nn.Parameter(torch.zeros(D))
        self.Wq = torch.nn.Parameter(torch.randn(da, H) / math.sqrt(H))
        self.Wk = torch.nn.Parameter(torch.randn(da, H) / math.sqrt(H))
        self.Wv = torch.nn.Parameter(torch.randn(dv, H) / math.sqrt(H))
        self.Wa = torch.nn.Parameter(torch.randn(D, dv) * (0.5 / math.sqrt(dv)))
        self.Ws = torch.nn.Parameter(torch.randn(D, D) * (0.5 / math.sqrt(D)))
        C = len(classes)
        self.Wc = torch.nn.Parameter(torch.randn(C, D) * 0.05)
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

    def run(self, x, is_u, is_msg, slot, is_dec, emb=None, drop=0.0, rdrop=0.0):
        """x: (B, T) tokens -> o (B, T, D): output vectors (meaningful at the is_dec positions)"""
        E = self.emb if emb is None else emb
        B, T = x.shape
        H = self.H
        xv = E[x]
        if drop > 0 and self.training:
            xv = torch.nn.functional.dropout(xv, drop)
        xe = xv @ self.Wx.t() + self.b
        Wh = self.Wh.t()
        h = xv.new_zeros(B, H)
        c = xv.new_zeros(B, H)
        zero = xv.new_zeros(B, self.Dv)
        Wkv = torch.cat([self.Wk, self.Wv], 0).t()             # keys and values of a message token state
        mem = xv.new_zeros(B, MAX_M, self.Da + self.Dv)
        filled = torch.zeros(B, MAX_M, dtype=torch.bool, device=x.device)
        onehot = torch.nn.functional.one_hot(slot, MAX_M).bool() & is_msg[:, :, None]
        hs, ctxs = [], []
        # recurrent dropout of the Millennium cell (RNNConfig.recurrentDropoutRate): one mask of h for all steps
        rmask = None
        if rdrop > 0 and self.training:
            rmask = torch.bernoulli(xv.new_full((B, H), 1.0 - rdrop)) / (1.0 - rdrop)
        for t in range(T):
            z = xe[:, t] + (h if rmask is None else h * rmask) @ Wh
            f = torch.sigmoid(z[:, :H].clamp(-50, 50))
            i = torch.sigmoid(z[:, H:2 * H].clamp(-50, 50))
            g = torch.tanh(z[:, 2 * H:3 * H].clamp(-50, 50))
            og = torch.sigmoid(z[:, 3 * H:].clamp(-50, 50))
            c = f * c + i * g
            pre = og * torch.tanh(c)
            mu = pre.mean(-1, keepdim=True)
            var = ((pre - mu) ** 2).mean(-1, keepdim=True)
            h = self.gam * (pre - mu) / torch.sqrt(var + 1e-5) + self.bet
            hs.append(h)
            ctx = zero
            u = is_u[:, t]
            if bool(u.any()):                                    # a new message: the memory starts again
                keep = (~u)[:, None]
                mem = mem * keep[:, :, None]
                filled = filled & keep
            oh = onehot[:, t]
            if bool(oh.any()):
                mem = mem + oh[:, :, None] * (h @ Wkv)[:, None, :]
                filled = filled | oh
            d = is_dec[:, t]
            if bool(d.any()):
                q = h @ self.Wq.t()
                sc = (mem[:, :, :self.Da] * q[:, None, :]).sum(-1).masked_fill(~filled, -1e30)
                a = torch.softmax(sc, -1) * filled
                ctx = (a[:, :, None] * mem[:, :, self.Da:]).sum(1) * d[:, None]
            ctxs.append(ctx)
        return torch.stack(hs, 1) @ self.Wp.t() + self.bp + xv @ self.Ws.t() + torch.stack(ctxs, 1) @ self.Wa.t()

    def loss(self, x, m, r, emb=None, drop=0.0, rdrop=0.0, smooth=0.0):
        """mean -log P of the tokens marked in m (position t predicts x[t + 1]); r = roles of x"""
        is_u, is_msg, slot = r
        dec = m[:, :-1].bool()
        o = self.run(x[:, :-1], is_u[:, :-1], is_msg[:, :-1], slot[:, :-1], dec, emb, drop, rdrop)
        o = o[dec]
        if drop > 0 and self.training:
            o = torch.nn.functional.dropout(o, drop)
        y = x[:, 1:][dec]
        cy = self.cls_of[y]
        sm = smooth if self.training else 0.0               # RNNConfig.labelSmoothing
        lc = torch.nn.functional.cross_entropy(o @ self.Wc.t() + self.bc, cy, reduction="sum", label_smoothing=sm)
        E = self.emb if emb is None else emb
        lw = o.new_zeros(())
        for c in torch.unique(cy).tolist():
            mem = self.members[c]
            if len(mem) == 1:
                continue
            idx = (cy == c).nonzero(as_tuple=True)[0]
            logits = o[idx] @ E[mem].t() + self.bw[mem]
            lw = lw + torch.nn.functional.cross_entropy(logits, self.pos_in[y[idx]], reduction="sum", label_smoothing=sm)
        return (lc + lw) / max(1, len(y)), len(y)


def log_probs(m, x, mk, r, emb, w_id):
    """-> per predicted position (in the order of mk[:, :-1].nonzero()): log P(target), log P([W])"""
    is_u, is_msg, slot = r
    dec = mk[:, :-1].bool()
    o = m.run(x[:, :-1], is_u[:, :-1], is_msg[:, :-1], slot[:, :-1], dec, emb)[dec]
    y = x[:, 1:][dec]
    E = m.emb if emb is None else emb
    lc = torch.log_softmax(o @ m.Wc.t() + m.bc, -1)
    cy = m.cls_of[y]
    lp = lc.gather(1, cy.clamp(min=0)[:, None])[:, 0]
    for c in torch.unique(cy).tolist():
        mem = m.members[c]
        if c < 0 or len(mem) == 1:
            continue
        idx = (cy == c).nonzero(as_tuple=True)[0]
        lw = torch.log_softmax(o[idx] @ E[mem].t() + m.bw[mem], -1)
        lp[idx] = lp[idx] + lw.gather(1, m.pos_in[y[idx]][:, None])[:, 0]
    cw = int(m.cls_of[w_id])
    pw = lc[:, cw]
    mem = m.members[cw]
    if len(mem) > 1:
        pw = pw + torch.log_softmax(o @ E[mem].t() + m.bw[mem], -1)[:, int(m.pos_in[w_id])]
    return lp, pw, dec


LEVELS = 127         # levels of the embedding codes on each side of zero (--emb-levels)


def emb_scale(emb):
    return float(emb.detach().std()) * CLIP / LEVELS


def quantize(emb, scale):
    """8-bit embeddings: value = scale * code, code = -127 .. 127 (one scale for the whole table)"""
    return torch.clamp(torch.round(emb / scale), -LEVELS, LEVELS) * scale


def batches(idx, lengths, bs, rng):
    """batches of similar length (less padding)"""
    idx = idx[np.argsort(lengths[idx] + rng.random_sample(len(idx)) * 4)]
    out = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    rng.shuffle(out)
    return out


class Data:
    def __init__(self, path, sp, dev):
        D = np.load(os.path.join(path, "chat_data.npz"))
        self.X, self.M, self.split = D["X"], D["M"], D["split"]
        self.L = (self.X != 1).sum(1)
        self.sp, self.dev = sp, dev

    def batch(self, b):
        T = int(self.L[b].max())
        X = self.X[b, :T]
        is_u, is_msg, slot = roles(X, self.sp)
        t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=self.dev)  # noqa: E731
        return t(X, torch.long), t(self.M[b, :T]), (t(is_u), t(is_msg), t(slot))


def evaluate(m, data, idx, emb=None, bs=512):
    m.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, mk, r = data.batch(idx[i:i + bs])
            l, k = m.loss(x, mk, r, emb)
            tot += float(l) * k
            n += k
    m.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rdrop", type=float, default=0.0, help="recurrent dropout (one mask of the state for all steps)")
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--wd", type=float, default=1e-6, help="weight decay (AdamW); above 1e-4: matrices only, not the embeddings")
    ap.add_argument("--emb-levels", type=int, default=127, help="embedding codes -N .. N in the quantized epochs (the big model is "
                                                                 "stored with the letters A .. Y: 12)")
    ap.add_argument("--emb-clip", type=float, default=5.0, help="the embedding codes cover this many standard deviations")
    ap.add_argument("--da", type=int, default=64, help="attention: size of the query / key")
    ap.add_argument("--dv", type=int, default=128, help="attention: size of the value")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda / cpu")
    ap.add_argument("--qat", type=int, default=1, help="last epochs with 8-bit embeddings in the forward pass")
    ap.add_argument("--init", help="checkpoint to fine-tune (its token classes and sizes are kept)")
    ap.add_argument("--limit", type=int, default=0, help="train on N random sequences only (quick tests)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from <out>/chat.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    dev = a.device
    meta = json.load(open(os.path.join(a.data, "chat_meta.json")))
    sp = meta["special"]
    data = Data(a.data, sp, dev)
    V = meta["n_vocab"] + len(sp)
    out_path = os.path.join(a.out, "chat.pt")
    rs = torch.load(out_path, map_location="cpu", weights_only=False) if a.resume and os.path.exists(out_path) else None
    if rs and rs.get("done"):
        print(f"{out_path}: already trained")
        return
    if rs and rs.get("resume", {}).get("epochs") != a.epochs:
        rs = None                                            # another schedule: start again
    ck = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    if rs or ck:
        classes, arch = (rs or ck)["classes"], (rs or ck)["arch"]
    else:
        classes = make_classes(data.X, data.M, V)
        arch = {"dim": a.dim, "hid": a.hidden, "da": a.da, "dv": a.dv}
    m = ChatLM(V, classes, **arch)
    if rs or ck:
        m.load_state_dict((rs or ck)["state"])
    m.to(dev)
    m.members = [t.to(dev) for t in m.members]
    n_par = sum(p.numel() for p in m.parameters())
    sizes = np.array([len(c) for c in classes])
    tgt = data.X[:, 1:][data.M[:, :-1] == 1]
    exp_size = float(sizes[m.cls_of.cpu().numpy()[tgt]].mean())
    print(f"device {dev}; parameters {n_par} (embeddings {m.emb.numel()}); LSTM {arch['hid']}, embedding {arch['dim']}; "
          f"{len(classes)} classes over {int(sizes.sum())} tokens, {exp_size:.1f} tokens scored per step on average", flush=True)
    tr = np.where(data.split == 0)[0]
    va = np.where(data.split == 1)[0]
    if ck:
        cls = m.cls_of.cpu().numpy()
        bad = ((data.M[:, :-1] == 1) & (cls[data.X[:, 1:]] < 0)).any(1)
        print(f"fine-tuning: {int(bad.sum())} sequences with tokens outside the classes left out")
        tr, va = tr[~bad[tr]], va[~bad[va]]
    if a.limit:
        tr = tr[np.random.RandomState(7).permutation(len(tr))[:a.limit]]
    if a.wd > 1e-4:
        decay = [p for n, p in m.named_parameters() if p.dim() == 2 and n != "emb"]
        rest = [p for n, p in m.named_parameters() if not (p.dim() == 2 and n != "emb")]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 1e-6}], lr=a.lr)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.03)
    os.makedirs(a.out, exist_ok=True)
    global CLIP, LEVELS
    CLIP, LEVELS = a.emb_clip, a.emb_levels
    scale = ck.get("scale") if ck else None                  # set when the quantized training starts, then kept
    if ck and ck.get("levels", 127) != a.emb_levels:
        scale = None                                         # another grid: its scale is set anew
    first = 0
    if rs:
        opt.load_state_dict(rs["resume"]["opt"])
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(dev)
        sched.load_state_dict(rs["resume"]["sched"])
        first, scale = rs["resume"]["epoch"], rs["qscale"]
        print(f"resuming after epoch {first}", flush=True)
    for ep in range(first, a.epochs):
        qat = ep >= a.epochs - a.qat or bool(ck)
        if qat and scale is None:
            scale = emb_scale(m.emb)
        t0 = time.time()
        tot, n = 0.0, 0
        bl = batches(tr, data.L, a.batch, rng)
        for bi, b in enumerate(bl):
            x, mk, r = data.batch(b)
            emb = None
            if qat:
                emb = m.emb + (quantize(m.emb.detach(), scale) - m.emb).detach()   # straight-through 8-bit embeddings
            l, k = m.loss(x, mk, r, emb, a.dropout, a.rdrop, a.smooth)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(l.detach()) * k
            n += k
            if bi % 200 == 0:
                el = time.time() - t0
                eta = el / (bi + 1) * (len(bl) - bi - 1)
                print(f"  epoch {ep + 1}/{a.epochs} batch {bi}/{len(bl)}: loss {tot / max(1, n):.3f} "
                      f"({el:.0f} s, epoch ends in ~{eta / 60:.0f} min)", flush=True)
        sc = scale if scale is not None else emb_scale(m.emb)
        vl, vq = evaluate(m, data, va), evaluate(m, data, va, quantize(m.emb.detach(), sc))
        print(f"epoch {ep + 1}/{a.epochs} train {tot / n:.3f} | val {vl:.3f} (ppl {math.exp(vl):.1f}), 8-bit {vq:.3f} "
              f"(ppl {math.exp(vq):.1f}) ({time.time() - t0:.0f} s)", flush=True)
        save = {"state": {k: v.cpu() for k, v in m.state_dict().items()}, "classes": classes, "arch": arch,
                "V": V, "special": sp, "scale": sc, "qscale": scale, "levels": LEVELS, "done": ep == a.epochs - 1}
        if not save["done"]:
            save["resume"] = {"epoch": ep + 1, "epochs": a.epochs, "opt": opt.state_dict(), "sched": sched.state_dict()}
        torch.save(save, out_path + ".part")
        os.replace(out_path + ".part", out_path)


if __name__ == "__main__":
    main()
