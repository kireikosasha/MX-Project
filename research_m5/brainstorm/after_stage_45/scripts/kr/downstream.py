import sys, random, time
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import torch
exec(open("turtle.py").read().split("def main():")[0])
import walk
from collections import namedtuple

def kinds_of(s, combo, m):
    """the story as a way of links for walk.Walk: kind = (symbol, heading at that moment), sign +1"""
    sid = {x: i for i, x in enumerate(SYM)}
    hmaps = [[(q + k) % m for q in range(m)] for k in range(m)] + [[v] * m for v in range(m)]
    h, out = 0, []
    for c in s:
        out.append((sid[c] * m + h, 1)); h = hmaps[combo[sid[c]]][h]
    return out

rng = random.Random(1)
per = int(sys.argv[1])
train = data(rng, per, 1, 4); aside = data(rng, 300, 5, 8)
m = 4
for name, combo in [("true heading cell (left +1, right +3, step/noise keep)", (1, 3, 0, 0)), ("a wrong heading cell (left +1, right +1)", (1, 1, 0, 0))]:
    t0 = time.time()
    taught = [(kinds_of(s, combo, m), 0, g) for s, g in train if s]
    A = namedtuple("A", "seed begin steps numbers")(0, 6, 300, 2)
    best = None
    for seed in range(6):
        torch.manual_seed(seed)
        model = walk.Walk(len(SYM) * m, 1, 9, seed, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for step in range(300):
            batch = rng.sample(taught, min(256, len(taught)))
            p = model([x[0] for x in batch], torch.tensor([0] * len(batch)))
            loss = -torch.log(p[torch.arange(len(batch)), torch.tensor([x[2] for x in batch])].clamp_min(1e-6)).mean()
            w = torch.softmax(model.adds, -1)
            loss = loss + 0.5 * (1 - step / 300) * (w * torch.log(w.clamp_min(1e-9))).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.harden(); model.polish(taught)
        fit = sum(model.hard(s, q) == g for s, q, g in taught) / len(taught)
        if best is None or fit > best[0]: best = (fit, model)
    model = best[1]
    ev = lambda ds: sum(model.hard(kinds_of(s, combo, m), 0) == g for s, g in ds) / len(ds)
    print(f"{name:55s} fit {100*best[0]:5.1f}  carry-over(5-8) {100*ev(aside):5.1f}  len20 {100*ev(data(rng,500,20,20)):5.1f}  len100 {100*ev(data(rng,500,100,100)):5.1f}  ({time.time()-t0:.0f}s)", flush=True)
