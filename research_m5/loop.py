#!/usr/bin/env python3
"""
A loop that does not drift: a recurrent core over an exact memory, against a looped transformer.

The task has no words. A memory of nodes: every node has a kind, a value (a small number) and three pointer fields.
A question is a start node and an opcode. An opcode means either "one step along a field" or "follow a field until
a node of such a kind, and say: the node come to / how many steps it took / the sum of the values gone through /
the greatest of them". Which field, which kind, what to say - the model is not told: the opcodes are dealt at
random and their meaning has to be learnt from the answers alone. Taught on answers at most 4 steps away; tried on
answers 8, 16, 32 and 40 steps away, with as many turns of the loop as it takes and many more.

Two models:

  machine   the state of the loop is where a pointer stands in the exact memory and what a register holds - softly:
            a distribution over (node, number). A turn of the loop is a read of the memory: what stands on a node
            of the kind waited for stays, the rest moves along the field, and the register takes the step in (not
            at all / one more / the value of the node / the greater). What is learnt is only the routing - of every
            opcode the field, the kind, what the register does and which of the two is said (four small tables),
            by the gradient of the answers through the turns of the loop; then it is made hard: a plain program on
            whole numbers. A turn carries the mass of the state over and neither makes nor loses any: the state
            cannot grow or fade, and the answer is a fixed point - more turns change nothing.
  looped    a transformer block gone through again and again (the recipe of the recurrent-depth models: the input
            is given anew at every turn, the state is normed after every turn, the number of turns is drawn at
            random in teaching). Its state is a free vector for every node.

    python loop.py --minutes 12
"""
import argparse
import time

import torch
from torch import nn

N, R, K, V = 48, 3, 4, 10
WHAT = ("node", "steps", "sum", "greatest")
OPS = R + R * K * len(WHAT)                                      # a step along a field; a field followed until a kind, and what is said
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def meaning(code):
    """an opcode before it is dealt -> the field, the kind waited for (K: none, one step), what is said"""
    if code < R:
        return code, K, 0
    code -= R
    return code // (K * len(WHAT)), code // len(WHAT) % K, code % len(WHAT)


def deal(batch, far, gen, table, exactly=None):
    """tasks -> the pointers [b, field, node], the kinds and the values [b, node], the start, the opcode as the model
    sees it, the answer (a node, or N + a number), what is asked, how many steps away the answer lies. far: at most so
    many steps away (exactly: just so many, for the opcodes that follow; an opcode of one step is one step away)"""
    f = torch.randint(N, (batch, R, N), generator=gen)
    kind = torch.randint(K, (batch, N), generator=gen)
    value = torch.randint(V, (batch, N), generator=gen)
    code = torch.randint(OPS, (batch,), generator=gen)
    step, rest = code < R, (code - R).clamp_min(0)
    field = torch.where(step, code, rest // (K * len(WHAT)))
    stop = rest // len(WHAT) % K
    what = torch.where(step, torch.zeros_like(code), rest % len(WHAT))
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
    gone = value.gather(1, path[:, 1:]) * (torch.arange(N - 1)[None] < d[:, None])       # the values of the nodes gone through
    number = torch.stack([d, d, gone.sum(1), gone.max(1).values], 1)[ar, what]
    answer = torch.where(what == 0, path[ar, d], N + number)
    return tuple(x.to(DEV) for x in (f, kind, value, path[:, 0], table[code], answer, what, d))


class Machine(nn.Module):
    """the loop over the exact memory; learnt: of every opcode the field it follows, the kind it waits for (or none),
    what the register does at a step, and which is said - the node or the register"""

    def __init__(self, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.field = nn.Parameter(0.1 * torch.randn((OPS, R), generator=gen))
        self.until = nn.Parameter(0.1 * torch.randn((OPS, K + 1), generator=gen))
        self.reg = nn.Parameter(0.1 * torch.randn((OPS, 4), generator=gen))
        self.out = nn.Parameter(0.1 * torch.randn((OPS, 2), generator=gen))

    def forward(self, f, kind, value, start, op, loops, size):
        """softly. The state: how much is believed of every (node the pointer stands on, number in the register), the
        numbers up to size -> of every answer how much it is believed [b, N + size], the mass of the state"""
        b = len(start)
        wf, wu, wr, wo = (torch.softmax(t[op], -1) for t in (self.field, self.until, self.reg, self.out))
        c = torch.arange(size, device=DEV)[None, None].expand(b, N, size)
        to = [(c + 1).clamp(max=size - 1), (c + value[:, :, None]).clamp(max=size - 1), torch.maximum(c, value[:, :, None])]
        along = [f[:, r][:, :, None].expand(b, N, size) for r in range(R)]
        p = torch.zeros(b, N, size, device=DEV)
        p[torch.arange(b), start, 0] = 1
        stay = (wu.gather(1, kind) + wu[:, K:])[:, :, None]      # of every node: does the pointer stay on it
        for i in range(loops):
            go = p if i == 0 else p * (1 - stay)                 # (the first turn always moves)
            moved = torch.zeros_like(p)
            for r in range(R):
                moved.scatter_add_(1, along[r], go * wf[:, r, None, None])
            new = moved * wr[:, 0, None, None]                   # the register at the node come to: as it was, one more,
            for k in range(3):                                   # the value of the node more, the greater of the two
                new = new + torch.zeros_like(p).scatter_add_(2, to[k], moved * wr[:, k + 1, None, None])
            p = new if i == 0 else p * stay + new                # the mass is carried over: none made, none lost
        return torch.cat([p.sum(2) * wo[:, :1], p.sum(1) * wo[:, 1:]], 1), p.sum((1, 2))

    def hard(self, f, kind, value, start, op, loops):
        """the routing made hard (the choice most believed in): a plain program over the memory, on whole numbers"""
        r, u, g, o = (t[op].argmax(-1) for t in (self.field, self.until, self.reg, self.out))
        ar = torch.arange(len(start), device=DEV)
        at, c = start.clone(), torch.zeros_like(start)
        for i in range(loops):
            move = (kind[ar, at] != u) & (u < K) if i else torch.ones_like(at, dtype=torch.bool)
            to = f[ar, r, at]
            v = value[ar, to]
            new = torch.stack([c, c + 1, c + v, torch.maximum(c, v)], 1)[ar, g]
            at, c = torch.where(move, to, at), torch.where(move, new, c)
        return torch.where(o == 0, at, N + c)


class Looped(nn.Module):
    """a transformer block gone through again and again, the input given anew at every turn, the state normed after it"""

    def __init__(self, numbers, width=96, heads=4, layers=2, part=24):
        super().__init__()
        self.node, self.kind, self.value, self.op = nn.Embedding(N, part), nn.Embedding(K, part), nn.Embedding(V, part), nn.Embedding(OPS, part)
        self.token = nn.Linear((3 + R) * part, width)            # a node: its name, its kind, its value, the names its fields point at
        self.ask = nn.Linear(2 * part, width)                    # the question: the name of the start, the opcode
        self.adapter = nn.Linear(2 * width, width)
        self.block = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True), layers,
                                           enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.say = nn.Linear(width, N + numbers)

    def forward(self, f, kind, value, start, op, loops, back=8, norms=None):
        own = self.node.weight[None].expand(len(start), -1, -1)
        x = self.token(torch.cat([own, self.kind(kind), self.value(value)] + [self.node(f[:, r]) for r in range(R)], -1))
        e = torch.cat([self.ask(torch.cat([self.node(start), self.op(op)], -1))[:, None], x], 1)
        s = torch.zeros_like(e)
        for i in range(loops):
            with torch.set_grad_enabled(self.training and loops - i <= back):      # (the gradient goes through the last turns only)
                s = self.norm(self.block(self.adapter(torch.cat([s, e], -1))))
            if norms is not None:
                norms.append(float((s - e).norm(dim=-1).mean()))
        return self.say(s[:, 0])


def right(model, gen, table, d, loops, n=600):
    """how often the answer so many steps away is given rightly with so many turns of the loop, %, by what is asked.
    model None: by one who always says the commonest answer to what is asked"""
    f, kind, value, start, op, answer, what, far = deal(n, d, gen, table, exactly=d)
    if model is None:
        return [100 * float(answer[(what == w) & (far == d)].bincount().max()) / max(1, int(((what == w) & (far == d)).sum())) for w in range(len(WHAT))]
    with torch.no_grad():
        got = model(f, kind, value, start, op, loops)
    got = got if got.dim() == 1 else got.argmax(-1)
    return [100 * float((got == answer)[(what == w) & (far == d)].float().mean()) for w in range(len(WHAT))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12, help="how long the looped transformer is taught")
    ap.add_argument("--far", type=int, default=4, help="in teaching the answer lies at most so many steps away")
    ap.add_argument("--width", type=int, default=96, help="of the looped transformer")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    gen = torch.Generator().manual_seed(a.seed)
    table = torch.randperm(OPS, generator=gen)                   # how the opcodes are dealt: not told to the models
    tries = (1, 2, 4, 8, 16, 32, 40)
    numbers = (V - 1) * max(tries) + 1

    machine = Machine(a.seed).to(DEV)
    opt, t0, size = torch.optim.Adam(machine.parameters(), lr=0.05), time.time(), (V - 1) * a.far + 2
    for step in range(1500):
        f, kind, value, start, op, answer, _, _ = deal(64, a.far, gen, table)
        p, _ = machine(f, kind, value, start, op, a.far + int(torch.randint(0, 5, (), generator=gen)), size)     # (more turns than needed, by chance)
        loss = -torch.log(p[torch.arange(64, device=DEV), answer].clamp_min(1e-6)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0 or step == 1499:
            print(f"machine, step {step}: loss {loss.item():.3f}", flush=True)
    learnt = 0
    for code in range(OPS):
        o, (fld, stop, what) = int(table[code]), meaning(code)
        learnt += int(machine.field[o].argmax() == fld and machine.until[o].argmax() == stop and machine.out[o].argmax() == (what > 0)
                      and (what == 0 or machine.reg[o].argmax() == what))
    print(f"machine: {sum(p.numel() for p in machine.parameters())} numbers learnt in {time.time() - t0:.0f} s; opcodes whose meaning is learnt rightly: {learnt} of {OPS}", flush=True)

    looped = Looped(numbers, a.width, layers=a.layers).to(DEV)
    opt, t0, step, seen = torch.optim.AdamW(looped.parameters(), lr=1e-3, weight_decay=0.01), time.time(), 0, 0.0
    while time.time() - t0 < 60 * a.minutes:
        spent = (time.time() - t0) / (60 * a.minutes)
        for group in opt.param_groups:                           # (the steps grow at first, then shrink)
            group["lr"] = 1e-3 * min(1.0, step / 300) * (1 - 0.8 * spent)
        far = min(a.far, 1 + int(2 * a.far * spent))             # from the near to the far: the first half of the time
        f, kind, value, start, op, answer, _, _ = deal(64, far, gen, table)
        out = looped(f, kind, value, start, op, far + 1 + int(torch.randint(0, 5, (), generator=gen)))      # (more turns than needed, by chance)
        loss = torch.nn.functional.cross_entropy(out, answer)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(looped.parameters(), 1.0)
        opt.step()
        step += 1
        seen = 0.98 * seen + 0.02 * float((out.argmax(-1) == answer).float().mean())
        if step % 500 == 0:
            print(f"looped, step {step}, {(time.time() - t0) / 60:.1f} min: answers at most {far} steps away, loss {loss.item():.3f}, right {100 * seen:.0f}%", flush=True)
    looped.eval()
    print(f"looped transformer: {sum(p.numel() for p in looped.parameters())} weights, {step} steps in {(time.time() - t0) / 60:.1f} min", flush=True)

    soft = lambda *x: machine(*x, numbers + 1)[0]
    rows_ = (("machine, routing made hard, 64 turns of the loop", lambda *x: machine.hard(*x), lambda d: 64, 600),
             ("machine, routing made hard, 1024 turns", lambda *x: machine.hard(*x), lambda d: 1024, 600),
             ("machine, routing left soft as learnt, 64 turns", soft, lambda d: 64, 100),
             ("looped transformer, 9 turns (as taught)", looped, lambda d: 9, 200),
             ("looped transformer, twice as many turns as steps", looped, lambda d: max(9, 2 * d), 200),
             ("looped transformer, 64 turns", looped, lambda d: 64, 200),
             ("always the commonest answer of the column", None, lambda d: 0, 2000))
    print(f"\nanswers right, %, by how many steps away the answer lies (taught: at most {a.far}); for every model four rows: what is asked")
    print("%-52s%-10s" % ("steps away", "") + "".join(f"{d:7d}" for d in tries))
    for label, model, loops, n in rows_:
        got = [right(model, gen, table, d, loops(d), n) for d in tries]
        for w, what in enumerate(WHAT):
            print("%-52s%-10s" % (label if w == 0 else "", what) + "".join(f"{g[w]:7.1f}" for g in got), flush=True)
    norms = []
    f, kind, value, start, op, answer, _, _ = deal(200, 4, gen, table)
    with torch.no_grad():
        looped(f, kind, value, start, op, 64, norms=norms)
        _, mass = machine(f, kind, value, start, op, 256, (V - 1) * 4 + 2)
    print(f"\nthe state of the looped transformer (how far it stands from the input, mean) after 1, 9, 32, 64 turns: {norms[0]:.1f}, {norms[8]:.1f}, {norms[31]:.1f}, {norms[63]:.1f}")
    print(f"the state of the machine after 256 turns, routing left soft: its mass {float(mass.min()):.4f} to {float(mass.max()):.4f} (begun at 1)")


if __name__ == "__main__":
    main()
