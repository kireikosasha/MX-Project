"""Rivals for cells.py on the same families: an LSTM and a transformer taught on the same short stories (and on forty
times more), tested on the same long ones. A rival counts only if it answers the stories it was taught on (at least 95 %
on fresh stories of the taught lengths)."""
import argparse, math, random, time
import torch
import torch.nn as nn
from cells import FAMILIES
from deletang import TASKS as DEEPMIND

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
    """alibi: no position vectors; each head looks back with a penalty growing with the distance (Press, Smith and
    Lewis 2022, made for inputs longer than those taught on), and only back (causal), the asking token last."""
    def __init__(self, nsym, nlab, width=128, layers=4, heads=4, alibi=False):
        super().__init__()
        self.alibi, self.heads = alibi, heads
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
        if self.alibi:  # (the padding is after the asking token, which looks only back: no padding mask is needed)
            n = L + 1
            i = torch.arange(n, device=x.device)
            dist = (i[:, None] - i[None, :]).float()
            slopes = torch.tensor([2 ** (-8 * (k + 1) / self.heads) for k in range(self.heads)], device=x.device)
            bias = -slopes[:, None, None] * dist[None]
            bias = bias.masked_fill(dist[None] < 0, float("-inf"))
            fast = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)  # (the fast path of evaluation does not take this mask rightly)
            h = self.body(self.emb(x), mask=bias.repeat(B, 1, 1))
            torch.backends.mha.set_fastpath_enabled(fast)
        else:
            h = self.emb(x) + self.pos(L + 1)
            h = self.body(h, src_key_padding_mask=pad)
        return self.out(h[torch.arange(B, device=x.device), lengths])


def batch(stories, sid, lid, nsym):
    L = max(len(s) for s, _ in stories)
    x = torch.full((len(stories), L), nsym, dtype=torch.long)
    for i, (s, _) in enumerate(stories):
        x[i, :len(s)] = torch.tensor([sid[c] for c in s])
    lengths = torch.tensor([len(s) for s, _ in stories])
    y = torch.tensor([lid.get(a, -1) for _, a in stories])  # an answer the rival has no class for is never right
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
    ap.add_argument("families", nargs="*", default=list(FAMILIES), help="families of cells.py or tasks of deletang.py")
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
    ap.add_argument("--watch", action="store_true", help="the watcher's protocol: the answer of every beginning of a story "
                    "taught on is taught too, and (as the learner's 40 questions) 40 more stories of length 8-32 per --teach")
    ap.add_argument("--strict", action="store_true", help="with --folder: also every story up to a length and long stories kept alive (worlds/strict.py)")
    ap.add_argument("--folder", help="sealed families (folder/NN-name/family.py with ALPHABET and answer) instead of the named ones")
    a = ap.parse_args()
    if a.folder:  # the families of a sealed pool, loaded as sealed_run.py loads them
        import os, sealed_run
        names = sorted(d for d in os.listdir(a.folder) if os.path.exists(os.path.join(a.folder, d, "family.py")))
        for d in names:
            syms, answer = sealed_run.load(os.path.join(a.folder, d, "family.py"))
            FAMILIES[d] = (syms, (lambda f: lambda w: str(f(list(w))))(answer))
        a.families = [f for f in a.families if f in names] or names  # (named ones of the folder, or all of it)
    torch.set_num_threads(a.threads)
    global DEVICE
    DEVICE = torch.device(a.device)
    for name in a.families:
        if name in DEEPMIND:  # a task of Deletang et al.: stories come from its own generator, not from any string
            gen = DEEPMIND[name][1]
            draw = lambda rng, n, lo, hi: [gen(rng, rng.randint(lo, hi)) for _ in range(n)]
            tr = random.Random(7)
            seen = draw(tr, 3000, 1, 20)
            syms = sorted({x for s, _ in seen for x in s})
            labels = sorted({y for _, y in seen})
            tests = {"41-500": draw(random.Random(1041), a.test, 41, 500)}
            fresh = draw(random.Random(99), a.test, 1, a.longest)
            make = lambda rng, n: draw(rng, n, 1, a.longest)
        else:
            syms, oracle = FAMILIES[name]
            tr = random.Random(7)
            labels = sorted({oracle([tr.choice(syms) for _ in range(tr.randint(1, 30))]) for _ in range(5000)})
            tests = {str(L): stories(syms, oracle, random.Random(1000 + L), a.test, L, L) for L in (20, 100, 500)}
            if a.strict:  # (worlds/strict.py: every story up to a length, and long stories kept away from a final answer)
                import itertools
                from worlds import strict
                short = {2: 14, 3: 9, 4: 8, 5: 7, 6: 7, 7: 6, 8: 6}.get(len(syms), 5)  # (as worlds/strict.py)
                tests["every<=%d" % short] = [(list(w), oracle(list(w))) for n in range(1, short + 1) for w in itertools.product(syms, repeat=n)]
                for L in (100, 500):
                    rs = random.Random(5)
                    tests[f"alive{L}"] = [(w, oracle(w)) for w in (strict.alive(oracle, syms, L, rs) for _ in range(60))]
            fresh = stories(syms, oracle, random.Random(99), a.test, 1, a.longest)
            make = lambda rng, n: stories(syms, oracle, rng, n, 1, a.longest)
        sid = {c: i for i, c in enumerate(syms)}
        lid = {l: i for i, l in enumerate(labels)}
        for kind in a.kinds:
            for times in a.times:
                rows = []
                for seed in range(a.seeds):
                    torch.manual_seed(seed)
                    rng = random.Random(seed)
                    teach = make(rng, a.teach * times)
                    if a.watch and name not in DEEPMIND:  # (every beginning, and stories as long as the learner asks)
                        teach += stories(syms, oracle, rng, 40 * times, 8, 32)
                        teach = list({tuple(w[:i]): (list(w[:i]), oracle(list(w[:i]))) for w, _ in teach
                                      for i in range(1, len(w) + 1)}.values())
                    model = (LSTM(len(syms), len(labels), a.width) if kind == "lstm" else
                             Transformer(len(syms), len(labels), a.width, a.layers, alibi=kind == "alibi")).to(DEVICE)
                    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                    t0 = time.time()
                    for step in range(a.steps):
                        part = rng.sample(teach, min(64, len(teach)))
                        x, l, y = batch(part, sid, lid, len(syms))
                        loss = nn.functional.cross_entropy(model(x, l), y, ignore_index=-1)
                        opt.zero_grad(); loss.backward(); opt.step()
                    taught = accuracy(model, fresh, sid, lid, len(syms))
                    long = [accuracy(model, tests[L], sid, lid, len(syms), part=1000 if L.startswith('every') else 50) for L in tests]
                    rows.append((taught, *long))
                    print(f"  {name:9s} {kind:11s} x{times:<3d} seed {seed}: taught lengths {taught:5.1f}  "
                          f"length {'/'.join(tests)}: {' '.join(f'{v:5.1f}' for v in long)}  ({time.time() - t0:.0f} s)", flush=True)
                m = [sum(r[i] for r in rows) / len(rows) for i in range(len(rows[0]))]
                print(f"{name:9s} {kind:11s} x{times:<3d} mean of {len(rows)}: taught lengths {m[0]:5.1f}  "
                      f"length {'/'.join(tests)}: {' '.join(f'{v:5.1f}' for v in m[1:])}", flush=True)


if __name__ == "__main__":
    main()
