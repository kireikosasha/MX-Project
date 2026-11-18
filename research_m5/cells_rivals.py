"""Rivals for cells.py on the same families: an LSTM and a transformer taught on the same short stories (and on forty
times more), tested on the same long ones. A rival counts only if it answers the stories it was taught on (at least 95 %
on fresh stories of the taught lengths)."""
import argparse, math, random, time
import torch
import torch.nn as nn
from cells import FAMILIES

DEVICE = torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, nsym, nlab, width=128):
        super().__init__()
        self.emb = nn.Embedding(nsym + 1, 32, padding_idx=nsym)
        self.rnn = nn.LSTM(32, width, batch_first=True)
        self.out = nn.Linear(width, nlab)

    def forward(self, x, lengths):
        h, _ = self.rnn(self.emb(x))
        return self.out(h[torch.arange(len(x), device=x.device), lengths - 1])


class Transformer(nn.Module):
    def __init__(self, nsym, nlab, width=128, layers=4, heads=4):
        super().__init__()
        self.emb = nn.Embedding(nsym + 2, width)  # +1 pad, +1 the token that asks
        self.ask = nsym + 1
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(width, nlab)
        self.width = width

    def pos(self, n):
        dev = self.out.weight.device
        p = torch.arange(n, device=dev).float()[:, None]
        i = torch.arange(0, self.width, 2, device=dev).float()
        ang = p / torch.pow(10000.0, i / self.width)
        pe = torch.zeros(n, self.width, device=dev)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
        return pe

    def forward(self, x, lengths):
        B, L = x.shape
        x = x.clone()
        x = torch.cat([x, torch.full((B, 1), self.ask - 1, device=x.device)], 1)  # room for the asking token
        x[torch.arange(B, device=x.device), lengths] = self.ask
        pad = torch.arange(L + 1, device=x.device)[None, :] > lengths[:, None]
        h = self.emb(x) + self.pos(L + 1)
        h = self.body(h, src_key_padding_mask=pad)
        return self.out(h[torch.arange(B, device=x.device), lengths])


def batch(stories, sid, lid, nsym):
    L = max(len(s) for s, _ in stories)
    x = torch.full((len(stories), L), nsym, dtype=torch.long)
    for i, (s, _) in enumerate(stories):
        x[i, :len(s)] = torch.tensor([sid[c] for c in s])
    lengths = torch.tensor([len(s) for s, _ in stories])
    y = torch.tensor([lid.get(a, 0) for _, a in stories])
    return x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)


def accuracy(model, stories, sid, lid, nsym, part=100):
    model.eval()
    right = 0
    with torch.no_grad():
        for i in range(0, len(stories), part):
            x, l, y = batch(stories[i:i + part], sid, lid, nsym)
            right += (model(x, l).argmax(-1) == y).sum().item()
    model.train()
    return 100 * right / len(stories)


def stories(syms, oracle, rng, n, lo, hi):
    out = []
    for _ in range(n):
        w = [rng.choice(syms) for _ in range(rng.randint(lo, hi))]
        out.append((w, oracle(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--teach", type=int, default=300)
    ap.add_argument("--times", type=int, nargs="*", default=[1, 40], help="multiples of --teach to teach on")
    ap.add_argument("--longest", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kinds", nargs="*", default=["lstm", "transformer"])
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global DEVICE
    DEVICE = torch.device(a.device)
    for name in a.families:
        syms, oracle = FAMILIES[name]
        sid = {c: i for i, c in enumerate(syms)}
        tr = random.Random(7)
        labels = sorted({oracle([tr.choice(syms) for _ in range(tr.randint(1, 30))]) for _ in range(5000)})
        lid = {l: i for i, l in enumerate(labels)}
        tests = {L: stories(syms, oracle, random.Random(1000 + L), a.test, L, L) for L in (20, 100, 500)}
        fresh = stories(syms, oracle, random.Random(99), a.test, 1, a.longest)
        for kind in a.kinds:
            for times in a.times:
                rows = []
                for seed in range(a.seeds):
                    torch.manual_seed(seed)
                    rng = random.Random(seed)
                    teach = stories(syms, oracle, rng, a.teach * times, 1, a.longest)
                    model = (LSTM(len(syms), len(labels), a.width) if kind == "lstm" else
                             Transformer(len(syms), len(labels), a.width, a.layers)).to(DEVICE)
                    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                    t0 = time.time()
                    for step in range(a.steps):
                        part = rng.sample(teach, min(64, len(teach)))
                        x, l, y = batch(part, sid, lid, len(syms))
                        loss = nn.functional.cross_entropy(model(x, l), y)
                        opt.zero_grad(); loss.backward(); opt.step()
                    taught = accuracy(model, fresh, sid, lid, len(syms))
                    long = [accuracy(model, tests[L], sid, lid, len(syms), part=50 if L == 500 else 100) for L in (20, 100, 500)]
                    rows.append((taught, *long))
                    print(f"  {name:9s} {kind:11s} x{times:<3d} seed {seed}: taught lengths {taught:5.1f}  "
                          f"length 20/100/500: {long[0]:5.1f} {long[1]:5.1f} {long[2]:5.1f}  ({time.time() - t0:.0f} s)", flush=True)
                m = [sum(r[i] for r in rows) / len(rows) for i in range(4)]
                print(f"{name:9s} {kind:11s} x{times:<3d} mean of {len(rows)}: taught lengths {m[0]:5.1f}  "
                      f"length 20/100/500: {m[1]:5.1f} {m[2]:5.1f} {m[3]:5.1f}", flush=True)


if __name__ == "__main__":
    main()
