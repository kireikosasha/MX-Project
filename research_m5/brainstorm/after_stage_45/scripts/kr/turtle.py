"""A two-cell cascade: a heading cell (Z_m) and a 2-number counter whose step DEPENDS on the heading (a wreath product).
Family: 'left', 'right', 'step', 'noise'; answer: the signs of (x, y) (9 classes, the readout of the project's register).
Compared: cascade with dependency; the same two cells side by side without dependency; the register alone (counter2)."""
import itertools, random, time, sys
import torch

OFF = list(itertools.product((-1, 0, 1), repeat=2))

def truth(s):
    h, x, y = 0, 0, 0
    for c in s:
        if c == "left": h = (h + 1) % 4
        elif c == "right": h = (h - 1) % 4
        elif c == "step":
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][h]; x += dx; y += dy
    return 3 * ((x > 0) - (x < 0) + 1) + ((y > 0) - (y < 0) + 1)

SYM = ["left", "right", "step", "noise"]

def data(rng, n, lo, hi):
    return [(s, truth(s)) for s in ([rng.choice(SYM) for _ in range(rng.randint(lo, hi))] for _ in range(n))]

def learn(train, m, depend, seed, steps=400):
    """m: states of the heading cell (0: none). The heading cell: per symbol a shift or a reset of Z_m.
    The counter: per (symbol, heading state if depend else symbol) one offset of {-1,0,1}^2."""
    torch.manual_seed(seed)
    L = max(len(s) for s, _ in train); W = 2 * L + 1
    sid = {x: i for i, x in enumerate(SYM)}
    mh = max(m, 1)
    hmaps = [[(s + k) % mh for s in range(mh)] for k in range(mh)] + [[v] * mh for v in range(mh)]
    H = torch.zeros(len(hmaps), mh, mh)
    for i, f in enumerate(hmaps):
        H[i, torch.arange(mh), torch.tensor(f)] = 1
    ha = torch.nn.Parameter(0.1 * torch.randn(len(SYM), len(hmaps)))
    ca = torch.nn.Parameter(0.1 * torch.randn(len(SYM), mh if depend else 1, len(OFF)))
    say = torch.nn.Parameter(0.1 * torch.randn(9, 9))
    opt = torch.optim.Adam([ha, ca, say], lr=0.1)
    seqs = torch.tensor([[sid[x] for x in s] + [-1] * (L - len(s)) for s, _ in train]); gold = torch.tensor([g for _, g in train])
    B = len(train)
    for t in range(steps):
        wh, wc = ha.softmax(-1), ca.softmax(-1)
        p = torch.zeros(B, mh, W, W); p[:, 0, L, L] = 1           # joint: heading x counter
        for i in range(L):
            on = seqs[:, i] >= 0; sy = seqs[:, i].clamp_min(0)
            # the counter moves first, by the heading BEFORE the symbol's own heading change (a cascade reads the upstream state)
            w = wc[sy]                                            # [B, mh or 1, 9]
            w = w.expand(B, mh, len(OFF))
            q = sum(w[:, :, j, None, None] * torch.roll(p, o, (2, 3)) for j, o in enumerate(OFF))
            T = torch.einsum("bc,cij->bij", wh[sy], H)            # then the heading
            q = torch.einsum("bixy,bij->bjxy", q, T)
            p = torch.where(on[:, None, None, None], q, p)
        c = p.sum(1)
        sx = torch.stack([c[:, :L].sum(1), c[:, L], c[:, L + 1:].sum(1)], 1)       # signs of x  [B,3,W]
        sxy = torch.stack([sx[:, :, :L].sum(2), sx[:, :, L], sx[:, :, L + 1:].sum(2)], 2).reshape(B, 9)
        got = sxy @ say.softmax(-1)
        loss = -torch.log(got[torch.arange(B), gold].clamp_min(1e-6)).mean()
        loss = loss + 0.3 * (1 - t / steps) * ((wh * wh.clamp_min(1e-9).log()).sum(-1).mean() + (wc * wc.clamp_min(1e-9).log()).sum(-1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    hh = [hmaps[k] for k in ha.argmax(-1).tolist()]
    cc = ca.argmax(-1).tolist()
    sy_ = say.argmax(-1).tolist()
    def run(s):
        h, x, y = 0, 0, 0
        for c in s:
            dx, dy = OFF[cc[sid[c]][h if depend else 0]]; x += dx; y += dy
            h = hh[sid[c]][h]
        return 3 * ((x > 0) - (x < 0) + 1) + ((y > 0) - (y < 0) + 1)
    def score():
        tb = {}
        for s, g in train:
            tb.setdefault(run(s), {}).setdefault(g, 0); tb[run(s)][g] += 1
        return sum(max(v.values()) for v in tb.values())
    for _ in range(6):                                            # every hard choice tried over, exactly (as walk.polish)
        before = score()
        for i in range(len(SYM)):
            if m:
                best = (score(), hh[i])
                for f in hmaps:
                    hh[i] = f; sc = score()
                    if sc > best[0]: best = (sc, f)
                hh[i] = best[1]
            for h in range(mh if depend else 1):
                best = (score(), cc[i][h])
                for j in range(len(OFF)):
                    cc[i][h] = j; sc = score()
                    if sc > best[0]: best = (sc, j)
                cc[i][h] = best[1]
        if score() == before: break
    # readout recounted exactly (sign class -> answer); here identity is expected but is learnt
    tb = {}
    for s, g in train:
        tb.setdefault(run(s), {}).setdefault(g, 0); tb[run(s)][g] += 1
    tb = {k: max(v, key=v.get) for k, v in tb.items()}
    return lambda s: tb.get(run(s))

def main():
    rng = random.Random(1)
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    train = data(rng, per, 1, 4); aside = data(rng, 300, 5, 8)
    for name, m, dep in [("register alone (2 numbers)", 0, False), ("heading Z4 beside counter, no dependency", 4, False),
                         ("cascade: heading Z4 -> counter", 4, True), ("cascade: heading Z3 -> counter", 3, True), ("cascade: heading Z6 -> counter", 6, True)]:
        t0 = time.time(); best = None
        for seed in range(8):
            f = learn(train, m, dep, seed)
            fit = sum(f(s) == g for s, g in train) / len(train); car = sum(f(s) == g for s, g in aside) / len(aside)
            if best is None or (fit, car) > best[:2]: best = (fit, car, f)
        test = {Lg: sum(best[2](s) == g for s, g in data(rng, 500, Lg, Lg)) / 500 for Lg in (20, 100)}
        print(f"{name:42s} fit {100*best[0]:5.1f}  carry-over(5-8) {100*best[1]:5.1f}  len20 {100*test[20]:5.1f}  len100 {100*test[100]:5.1f}  ({time.time()-t0:.0f}s)", flush=True)

main()
