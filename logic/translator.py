#!/usr/bin/env python3
"""
The translator: a sentence -> a frame (predicate + the role of every word). The only learned part.

Network: word ids -> embedding -> bidirectional LSTM with the Millennium cell (gates, cell, LayerNorm of
o * tanh(c)) -> the role of every word (linear), and the predicate from the attention-pooled states.
It is small on purpose (about 0.1 M parameters): it does not reason, it only reads structure.

Decoding is constrained by the grammar of frames: for every predicate only its roles are allowed, the best
tagging under that restriction is scored, and a frame that lacks a role it needs is not accepted at all.

    python translator.py --out translator.pt            trains on generated sentences (CPU, a few minutes)
"""
import argparse
import math
import random

import torch

import world as W
from engine import MAY, NEEDS, PREDICATES, ROLES


class MillenniumLSTM(torch.nn.Module):
    """one direction of the Millennium LSTM layer: z = [x | h], gates, c = f c + i g, h = LayerNorm(o tanh(c))"""

    def __init__(self, n_in, hid):
        super().__init__()
        self.H = hid
        s = math.sqrt(6.0 / (n_in + hid))
        self.Wx = torch.nn.Parameter(torch.empty(4 * hid, n_in).uniform_(-s, s))
        self.Wh = torch.nn.Parameter(torch.empty(4 * hid, hid).uniform_(-1 / math.sqrt(hid), 1 / math.sqrt(hid)))
        b = torch.zeros(4, hid)
        b[0] = 1.0
        self.b = torch.nn.Parameter(b.reshape(-1))
        self.gam = torch.nn.Parameter(torch.ones(hid))
        self.bet = torch.nn.Parameter(torch.zeros(hid))

    def forward(self, x, mask, reverse=False):
        """x: (B, T, n_in), mask: (B, T) -> states (B, T, H); padded steps keep the state"""
        B, T, _ = x.shape
        H = self.H
        xe = x @ self.Wx.t() + self.b
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        out = [None] * T
        for t in (range(T - 1, -1, -1) if reverse else range(T)):
            z = xe[:, t] + h @ self.Wh.t()
            f, i = torch.sigmoid(z[:, :H]), torch.sigmoid(z[:, H:2 * H])
            g, o = torch.tanh(z[:, 2 * H:3 * H]), torch.sigmoid(z[:, 3 * H:])
            c2 = f * c + i * g
            pre = o * torch.tanh(c2)
            mu = pre.mean(-1, keepdim=True)
            var = ((pre - mu) ** 2).mean(-1, keepdim=True)
            h2 = self.gam * (pre - mu) / torch.sqrt(var + 1e-5) + self.bet
            m = mask[:, t:t + 1]
            h, c = m * h2 + (1 - m) * h, m * c2 + (1 - m) * c
            out[t] = h
        return torch.stack(out, 1)


class Translator(torch.nn.Module):
    def __init__(self, dim=48, hid=64):
        super().__init__()
        self.emb = torch.nn.Embedding(W.N_VOCAB, dim, padding_idx=0)
        self.fw, self.bw = MillenniumLSTM(dim, hid), MillenniumLSTM(dim, hid)
        self.role = torch.nn.Linear(2 * hid, len(ROLES))
        self.att = torch.nn.Linear(2 * hid, 1)
        self.pred = torch.nn.Linear(2 * hid, len(PREDICATES))

    def forward(self, ids):
        mask = (ids != 0).float()
        x = self.emb(ids)
        h = torch.cat([self.fw(x, mask), self.bw(x, mask, reverse=True)], -1)
        a = torch.softmax(self.att(h).squeeze(-1).masked_fill(mask == 0, -1e30), -1)        # attention pooling
        return self.pred((a.unsqueeze(-1) * h).sum(1)), self.role(h)

    # ---- sentence -> frame
    def parse(self, text):
        words = W.tokens(text)
        if not words:
            return None
        with torch.no_grad():
            lp, lr = self(torch.tensor([[W.word_id(w) for w in words]]))
        lp, lr = torch.log_softmax(lp[0], -1), torch.log_softmax(lr[0], -1)
        best = None
        for pi, p in enumerate(PREDICATES):                 # the grammar: only the roles of this predicate
            ok = [ROLES.index(r) for r in ("O",) + NEEDS[p] + MAY.get(p, ())]
            sub = lr[:, ok]
            score, pick = sub.max(-1)
            tags = [ROLES[ok[k]] for k in pick.tolist()]
            if not all(r in tags for r in NEEDS[p]):
                continue                                     # a role it needs is missing: not a frame
            total = float(lp[pi]) + float(score.sum())
            if best is None or total > best[0]:
                best = (total, p, tags)
        if best is None:
            return None
        return W.frame(words, best[1], best[2])


def batch(rng, n):
    rows = [W.sentence(rng) for _ in range(n)]
    T = max(len(w) for w, _, _ in rows)
    ids = torch.zeros(n, T, dtype=torch.long)
    roles = torch.full((n, T), -100, dtype=torch.long)
    for i, (w, _, r) in enumerate(rows):
        ids[i, :len(w)] = torch.tensor([W.word_id(x) for x in w])
        roles[i, :len(w)] = torch.tensor([ROLES.index(x) for x in r])
    return ids, torch.tensor([PREDICATES.index(p) for _, p, _ in rows]), roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="translator.pt")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    m = Translator(a.dim, a.hid)
    print("parameters:", sum(p.numel() for p in m.parameters()))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.steps, pct_start=0.05)
    ce = torch.nn.functional.cross_entropy
    for step in range(a.steps):
        ids, pred, roles = batch(rng, a.batch)
        lp, lr = m(ids)
        loss = ce(lp, pred) + ce(lr.reshape(-1, len(ROLES)), roles.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            print(f"step {step}: loss {float(loss):.4f}", flush=True)
    m.eval()
    test = random.Random(a.seed + 1000)                      # new made-up words: frames must come out exactly
    right, n = 0, 2000
    for _ in range(n):
        w, p, r = W.sentence(test)
        right += m.parse(" ".join(w)) == W.frame(w, p, r)
    print(f"fresh sentences: {right}/{n} frames exactly right")
    torch.save({"state": m.state_dict(), "dim": a.dim, "hid": a.hid}, a.out)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Translator(ck["dim"], ck["hid"])
    m.load_state_dict(ck["state"])
    return m.eval()


if __name__ == "__main__":
    main()
