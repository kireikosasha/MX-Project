"""One fold machine with three parts, all learnt by the gradient of the answers through the fold, then made hard:
   Z   an integer counter: a symbol adds one of -2..2           (the register of walk.py, magnitude kept)
   Q   a finite control of k states: a symbol moves it by a learnt table (the table of walk.py, as an automaton)
   R   a latest-value cell of m values: a symbol sets it or keeps it  (the overwrite of recall.py)
   SAY a table (clip(z), q, r) -> answer word.
A symbol is (kind of link, direction). register = (Z, k=1, m=1); table = (k=|R|); recall's value = R.
CLUTRR with the links given, taught on N stories of 2-3 links, tried on 2..10 links."""
import sys, random, itertools, time
import torch
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import walk
D, N, K, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
SEEDS = int(sys.argv[5]) if len(sys.argv) > 5 else 6
torch.set_num_threads(2)
kinds, asks, answers = {}, {}, {}
taught = walk.kin(f"{D}/gen23_train.csv", kinds, asks, answers, True)
test = walk.kin(f"{D}/gen23_test.csv", kinds, asks, answers, False)
sym = lambda k, s: 2 * k + (s < 0)
S, A, OFF, Z = 2 * len(kinds), len(answers), [-2, -1, 0, 1, 2], 7       # z clipped to -3..3 for SAY; the fold itself is exact
rng = random.Random(0)
data = [(st, g) for st, q, g, _ in rng.sample(taught, N) if st]
def soft(params, ways):
    phi, tq, tr, say = params
    far = max(len(w) for w in ways); B = len(ways); W = 2 * 2 * far + 1
    pz = torch.zeros(B, W); pz[:, W // 2] = 1
    pq = torch.zeros(B, K); pq[:, 0] = 1
    pr = torch.zeros(B, M + 0); pr[:, 0] = 1
    for i in range(far):
        on = torch.tensor([i < len(w) for w in ways])
        s = torch.tensor([sym(*w[i]) if i < len(w) else 0 for w in ways])
        a = torch.nn.functional.one_hot(torch.tensor([FIX[x] for x in s.tolist()]), 5).float()
        nz = sum(a[:, j:j + 1] * torch.roll(pz, o, 1) for j, o in enumerate(OFF))
        nq = torch.einsum("bk,bkl->bl", pq, tq[s].softmax(-1))         # [B,K] x [B,K,K]
        w = tr[s].softmax(-1)                                           # [B, M+1]: set to value j, or keep (last)
        nr = w[:, :M] + w[:, M:] * pr
        pz, pq, pr = (torch.where(on[:, None], n, p) for n, p in ((nz, pz), (nq, pq), (nr, pr)))
    c = W // 2
    zc = torch.stack([pz[:, :c - 2].sum(1)] + [pz[:, c + d] for d in (-2, -1, 0, 1, 2)] + [pz[:, c + 3:].sum(1)], 1)   # clip to 7 bins
    joint = torch.einsum("bz,bq,br->bzqr", zc, pq, pr).reshape(B, -1)
    return joint @ say.softmax(-1)
def hard(h, way):
    phi, tq, tr, say = h
    z, q, r = 0, 0, 0
    for k, s in way:
        x = sym(k, s); z += OFF[phi[x]]; q = tq[x][q]; r = tr[x] if tr[x] < M else r
    return say[(max(-3, min(3, z)) + 3) * K * M + q * M + r]
def harden(p):
    phi, tq, tr, say = p
    return [phi.argmax(-1).tolist(), tq.argmax(-1).tolist(), tr.argmax(-1).tolist(), say.argmax(-1).tolist()]
def score(h, d): return sum(hard(h, w) == g for w, g in d)
def polish(h, d, rounds=3):
    for _ in range(rounds):
        moved = False
        cells = Counter = {}
        # SAY: the commonest answer of every hard state
        from collections import Counter as C
        seen = {}
        for w, g in d:
            phi, tq, tr, say = h; z, q, r = 0, 0, 0
            for k, s in w:
                x = sym(k, s); z += OFF[phi[x]]; q = tq[x][q]; r = tr[x] if tr[x] < M else r
            seen.setdefault((max(-3, min(3, z)) + 3) * K * M + q * M + r, C())[g] += 1
        for cell, c in seen.items():
            if h[3][cell] != c.most_common(1)[0][0]: h[3][cell] = c.most_common(1)[0][0]; moved = True
        for part, size in ((2, M + 1),):
            for x in range(S):
                was, best = h[part][x], (score(h, d), h[part][x])
                for j in range(size):
                    h[part][x] = j; sc = score(h, d)
                    if sc > best[0]: best = (sc, j)
                h[part][x] = best[1]; moved |= best[1] != was
        for x in range(S):
            for q0 in range(K):
                was, best = h[1][x][q0], (score(h, d), h[1][x][q0])
                for j in range(K):
                    h[1][x][q0] = j; sc = score(h, d)
                    if sc > best[0]: best = (sc, j)
                h[1][x][q0] = best[1]; moved |= best[1] != was
        if not moved: break
    return h
import numpy as np
names = {i: k for k, i in kinds.items()}
ans_name = {i: k for k, i in answers.items()}
rel = sorted(set(kinds) | set(answers)); jx = {r: i for i, r in enumerate(rel)}
Aeq = []
for w, g in data:
    v = np.zeros(len(rel)); v[jx[ans_name[g]]] += 1
    for k, s in w: v[jx[names[k]]] -= s
    Aeq.append(v)
u, sv, vt = np.linalg.svd(np.array(Aeq), full_matrices=True)
null = vt[int(np.sum(sv > 1e-9)):]
print("null space dimension", len(null))
gr = null[0] / null[0][np.argmax(np.abs(null[0]))] * 2
gr = np.round(gr).astype(int)
FIX = [OFF.index(int(gr[jx[names[x // 2]]]) * (1 if x % 2 == 0 else -1)) if names.get(x // 2) in jx else 2 for x in range(S)]
print("found grading:", {names[i]: int(gr[jx[names[i]]]) for i in names})
t0, best = time.time(), None
for seed in range(SEEDS):
    torch.manual_seed(seed)
    p = [torch.nn.Parameter(0.1 * torch.randn(*sh)) for sh in ((S, 5), (S, K, K), (S, M + 1), (Z * K * M, A))]
    opt = torch.optim.Adam(p, lr=0.1)
    steps = 400
    for step in range(steps):
        batch = data if len(data) <= 256 else random.sample(data, 256)
        out = soft(p, [w for w, _ in batch])
        loss = -torch.log(out[torch.arange(len(batch)), torch.tensor([g for _, g in batch])].clamp_min(1e-6)).mean()
        ent = sum((t.softmax(-1) * t.log_softmax(-1)).sum(-1).mean() for t in p[:3])
        loss = loss + 0.3 * (1 - step / steps) * ent
        opt.zero_grad(); loss.backward(); opt.step()
    h = harden([x.detach() for x in p]); h[0] = list(FIX); h = polish(h, data)
    sc = score(h, data) / len(data)
    print(f"  begun from {seed}: explains {100*sc:.1f}% of the {len(data)} stories taught on")
    if best is None or sc > best[0]: best = (sc, h)
    if sc == 1: break
h = best[1]
by = {}
for st, q, g, k in test:
    by.setdefault(k, []).append(st is not None and hard(h, st) == g)
print(f"N={N} K={K} M={M}: test by links " + " ".join(f"{k}:{100*sum(v)/len(v):5.1f}" for k, v in sorted(by.items())) + f"  all {100*sum(map(sum, by.values()))/len(test):.1f}  ({time.time()-t0:.0f} s)")
print("what each link adds to the counter:", {k: OFF[h[0][sym(i, 1)]] for k, i in kinds.items()})
