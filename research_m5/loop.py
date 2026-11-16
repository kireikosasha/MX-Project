#!/usr/bin/env python3
"""
A loop that does not drift: a recurrent core over an exact memory, against a looped transformer.

The task has no words. A memory of nodes: every node has a kind and three pointer fields. A question is a start
node and an opcode; an opcode means either "one step along a field" or "follow a field until a node of such a kind"
(which field, which kind - the model is not told: the opcodes are dealt at random and their meaning has to be
learnt from the answers alone). The answer lies so many steps from the start. Taught on answers at most 4 steps
away; tried on answers 8, 16, 32 and 40 steps away, with as many turns of the loop as it takes and many more.

Two models:

  machine   the state of the loop is a pointer into the exact memory (a distribution over the nodes). A turn of the
            loop is a read of the memory: what stands at the pointer stays if its kind is the one waited for, and
            moves along the field otherwise. What is learnt is only the routing - for every opcode, which field and
            which kind (two small tables), by the gradient of the answers through the turns of the loop, softly;
            then it is made hard. A turn carries the mass of the pointer over and neither makes nor loses any: the
            state cannot grow or fade, and the answer is a fixed point - more turns change nothing.
  looped    a transformer block gone through again and again (the recipe of the recurrent-depth models: the input
            is given anew at every turn, the state begins at random, the number of turns is drawn at random in
            teaching). Its state is a free vector for every node.

    python loop.py --minutes 12
"""
import argparse
import time

import torch
from torch import nn

N, R, K = 48, 3, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OPS = R + R * K                                                  # a step along a field; a field followed until a kind


def deal(batch, far, gen, table, exactly=None):
    """tasks -> the pointers [b, field, node], the kinds [b, node], the start, the opcode as the model sees it, the
    answer, how far it lies. far: at most so many steps (exactly: just so many, for the opcodes that follow)"""
    f = torch.randint(N, (batch, R, N), generator=gen)
    kind = torch.randint(K, (batch, N), generator=gen)
    code = torch.randint(OPS, (batch,), generator=gen)
    step = code < R
    field = torch.where(step, code, (code - R) // K)
    stop = (code - R) % K
    d = torch.randint(1, far + 1, (batch,), generator=gen) if exactly is None else torch.full((batch,), exactly)
    d = torch.where(step, torch.ones_like(d), d)
    path = torch.argsort(torch.rand((batch, N), generator=gen), 1)       # nodes all different: the way from the start
    ar = torch.arange(batch)
    for i in range(int(d.max())):
        on = i < d
        f[ar[on], field[on], path[on, i]] = path[on, i + 1]
        mid = on & (i + 1 < d) & ~step                           # a node on the way is not of the kind waited for
        other = (stop + 1 + torch.randint(K - 1, (batch,), generator=gen)) % K
        kind[ar[mid], path[mid, i + 1]] = other[mid]
    last = ~step
    kind[ar[last], path[last, d[last]]] = stop[last]
    return tuple(x.to(DEV) for x in (f, kind, path[:, 0], table[code], path[ar, d], d))


class Machine(nn.Module):
    """the loop over the exact memory; learnt: of every opcode the field it follows and the kind it waits for (or none)"""

    def __init__(self, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.field = nn.Parameter(0.1 * torch.randn((OPS, R), generator=gen))
        self.until = nn.Parameter(0.1 * torch.randn((OPS, K + 1), generator=gen))

    def forward(self, f, kind, start, op, loops, hard=False):
        wf, wu = torch.softmax(self.field[op], -1), torch.softmax(self.until[op], -1)
        if hard:                                                 # the routing made hard: the choice most believed in
            wf = torch.nn.functional.one_hot(wf.argmax(-1), R).float()
            wu = torch.nn.functional.one_hot(wu.argmax(-1), K + 1).float()
        p = torch.nn.functional.one_hot(start, N).float()
        stay = wu.gather(1, kind) + wu[:, K:]                    # of every node: does the pointer stay on it
        for i in range(loops):
            go = p if i == 0 else p * (1 - stay)                 # (the first turn always moves)
            moved = torch.zeros_like(p)
            for r in range(R):
                moved.scatter_add_(1, f[:, r], go * wf[:, r:r + 1])
            p = moved if i == 0 else p * stay + moved            # the mass is carried over: none made, none lost
        return p


class Looped(nn.Module):
    """a transformer block gone through again and again, the input given anew at every turn"""

    def __init__(self, width=96, heads=4, layers=2):
        super().__init__()
        self.node, self.kind = nn.Embedding(N, width), nn.Embedding(K, width)        # (one table of the names of the nodes: a pointer
        self.ptr = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(R)])   # is the name of its node, turned by its field)
        self.start, self.op = nn.Linear(width, width, bias=False), nn.Embedding(OPS, width)
        self.adapter = nn.Linear(2 * width, width)
        self.block = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True), layers,
                                           enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)                          # (the answer is said by the name of its node: the same table)

    def forward(self, f, kind, start, op, loops, norms=None):
        x = self.node.weight[None] + self.kind(kind) + sum(self.ptr[r](self.node(f[:, r])) for r in range(R))
        e = torch.cat([(self.start(self.node(start)) + self.op(op))[:, None], x], 1)
        s = (0.4 * torch.randn_like(e)) if self.training else torch.zeros_like(e)       # (in teaching the state begins at random)
        for i in range(loops):
            s = self.block(self.adapter(torch.cat([s, e], -1)))
            if norms is not None:
                norms.append(float(s.norm(dim=-1).mean()))
        return self.norm(s[:, 0]) @ self.node.weight.T


def right(model, gen, table, d, loops, n=600, **more):
    """how often the answer so many steps away is given rightly with so many turns of the loop, %"""
    f, kind, start, op, answer, _ = deal(n, d, gen, table, exactly=d)
    with torch.no_grad():
        out = model(f, kind, start, op, loops, **more)
    return 100 * float((out.argmax(-1) == answer).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12, help="how long the looped transformer is taught")
    ap.add_argument("--far", type=int, default=4, help="in teaching the answer lies at most so many steps away")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(a.seed)
    table = torch.randperm(OPS, generator=gen)                   # how the opcodes are dealt: not told to the models
    tries = (1, 2, 4, 8, 16, 32, 40)

    machine = Machine(a.seed).to(DEV)
    opt, t0 = torch.optim.Adam(machine.parameters(), lr=0.05), time.time()
    for step in range(1500):
        f, kind, start, op, answer, d = deal(64, a.far, gen, table)
        p = machine(f, kind, start, op, a.far + int(torch.randint(0, 9, (), generator=gen)))      # (more turns than needed, by chance)
        loss = -torch.log(p[torch.arange(64, device=DEV), answer].clamp_min(1e-6)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0 or step == 1499:
            print(f"machine, step {step}: loss {loss.item():.3f}", flush=True)
    field, until = machine.field.detach().cpu(), machine.until.detach().cpu()
    learnt = sum((field[table[c]].argmax() == (c if c < R else (c - R) // K)) and (until[table[c]].argmax() == (K if c < R else (c - R) % K)) for c in range(OPS))
    print(f"machine: {sum(p.numel() for p in machine.parameters())} numbers learnt in {time.time() - t0:.0f} s; opcodes whose field and kind are learnt rightly: {int(learnt)} of {OPS}", flush=True)

    looped = Looped().to(DEV)
    opt, t0, step = torch.optim.Adam(looped.parameters(), lr=1e-3), time.time(), 0
    while time.time() - t0 < 60 * a.minutes:
        spent = (time.time() - t0) / (60 * a.minutes)
        for group in opt.param_groups:                           # (the steps grow at first, then shrink)
            group["lr"] = 1e-3 * min(1.0, step / 200) * (1 - 0.8 * spent)
        f, kind, start, op, answer, d = deal(48, min(a.far, 1 + int(2 * a.far * spent)), gen, table)       # from the near to the far: the first half of the time
        out = looped(f, kind, start, op, int(torch.randint(6, 13, (), generator=gen)))             # (the number of turns drawn at random)
        loss = torch.nn.functional.cross_entropy(out, answer)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(looped.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            print(f"looped, step {step}, {(time.time() - t0) / 60:.1f} min: loss {loss.item():.3f}", flush=True)
    print(f"looped transformer: {sum(p.numel() for p in looped.parameters())} weights, {step} steps in {(time.time() - t0) / 60:.1f} min", flush=True)

    print(f"\nanswers right, %, by how many steps away the answer lies (taught: at most {a.far}; chance: {100 / N:.0f}%)")
    print("%-58s" % "steps away" + "".join(f"{d:7d}" for d in tries))
    rows_ = (("machine, routing made hard, 64 turns of the loop", machine, lambda d: 64, {"hard": True}),
             ("machine, routing made hard, 1024 turns", machine, lambda d: 1024, {"hard": True}),
             ("machine, routing left soft as learnt, 64 turns", machine, lambda d: 64, {}),
             ("looped transformer, 9 turns (as taught)", looped, lambda d: 9, {}),
             ("looped transformer, twice as many turns as steps", looped, lambda d: max(9, 2 * d), {}),
             ("looped transformer, 64 turns", looped, lambda d: 64, {}))
    for label, model, loops, more in rows_:
        print("%-58s" % label + "".join(f"{right(model, gen, table, d, loops(d), **more):7.1f}" for d in tries), flush=True)
    norms = []
    f, kind, start, op, answer, _ = deal(200, 4, gen, table)
    with torch.no_grad():
        looped(f, kind, start, op, 64, norms=norms)
        p = machine(f, kind, start, op, 1024, hard=True)
    print(f"\nthe state of the looped transformer (mean length of a vector) after 1, 9, 32, 64 turns: {norms[0]:.1f}, {norms[8]:.1f}, {norms[31]:.1f}, {norms[63]:.1f}")
    print(f"the state of the machine after 1024 turns: the mass of the pointer {float(p.sum(-1).min()):.4f} to {float(p.sum(-1).max()):.4f} (begun at 1)")


if __name__ == "__main__":
    main()
