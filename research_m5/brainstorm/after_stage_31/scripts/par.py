"""soft.py's sequential replay of the value memory == stick-breaking attention over writes with exact (identity)
keys, plus a few loop iterations for the parts whose content depends on the state (as, lift, swap).
Checks equality on real tasks and times both (forward + backward)."""
import sys, glob, os, time
sys.path.insert(0, "/home/user/MX-Project/research_m5")
sys.dont_write_bytecode = True
import torch
import recall, rows, soft
from soft import VALUE, SLOTS

L = "/tmp/claude-0/-home-user-MX-Project/ace7a071-a9eb-520e-b8a4-5dce652d6be1/scratchpad/logic"


def load(folder, k, n=40):
    raw = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], n)
    for reader in rows.guesses(raw[:20]):
        task = recall.Task(reader.rows(raw), False)
        if max(sum(x != "#" for x in f[2]) for f in list(task.forms) + list(task.asked)) <= SLOTS:
            return task
    return None


def sequential(data, pv, pa):
    """as in soft.Layer.forward (the value memory only) -> V before every sub-step: (B, 2S+1, N, N+1)"""
    B, S, N = data.said.shape[0], data.said.shape[1], data.N
    V = torch.zeros((B, N, N + 1), dtype=pv.dtype); V[:, :, N] = 1.0
    ar, none = torch.arange(B), torch.nn.functional.one_hot(torch.tensor(N), N + 1).to(pv.dtype)
    hist = [V]
    for s in range(S):
        f, t = data.said[:, s], data.things[:, s]
        for p in (pv[f], pa[f]):
            D = torch.zeros_like(V)
            for o, part in enumerate(VALUE):
                if part is None:
                    continue
                rows_ = [t[:, part[1]]]
                cur = V[ar, rows_[0]]
                if part[0] == "set":
                    new = [torch.nn.functional.one_hot(t[:, part[2]], N + 1).to(pv.dtype)]
                elif part[0] == "clear":
                    new = [none.expand(B, N + 1)]
                elif part[0] == "as":
                    new = [V[ar, t[:, part[2]]]]
                elif part[0] == "lift":
                    up = torch.bmm(cur[:, None, :N], V)[:, 0]
                    new = [up + cur[:, N:] * none]
                else:
                    rows_.append(t[:, part[2]])
                    new = [V[ar, rows_[1]], cur]
                for r, x in zip(rows_, new):
                    D[ar, r] = D[ar, r] + p[:, o, None] * (x - V[ar, r])
            V = V + D
            hist.append(V)
    return torch.stack(hist, 1)


def parallel(data, pv, pa, loops, ref=None, tol=1e-6):
    """the same with no loop over the story: every write is (a row, how much, a content); the state before a sub-step
    is a stick-breaking sum over the writes before it; contents that read the state are refreshed `loops` times.
    -> V before every sub-step, and after how many loops it was equal to ref"""
    B, S, N = data.said.shape[0], data.said.shape[1], data.N
    T, dt = 2 * S, pv.dtype
    p = torch.stack([pv[data.said], pa[data.said]], 2).reshape(B, T, len(VALUE))            # (B, T, O)
    t = data.things[:, :, None, :].expand(B, S, 2, SLOTS).reshape(B, T, SLOTS)
    eye, eye1 = torch.eye(N, dtype=dt), torch.eye(N + 1, dtype=dt)
    none = eye1[N]
    hot = eye[t]                                                # (B, T, SLOTS, N): which row a slot names
    m = torch.zeros((B, T, N), dtype=dt)                        # how much of every row a sub-step writes over
    for o, part in enumerate(VALUE):
        if part is None:
            continue
        m = m + p[:, :, o, None] * hot[:, :, part[1]]
        if part[0] == "swap":
            m = m + p[:, :, o, None] * hot[:, :, part[2]]
    keep = torch.log((1 - m).clamp(min=1e-30))
    Lc = torch.cat([torch.zeros((B, 1, N), dtype=dt), keep.cumsum(1)], 1)                   # (B, T+1, N)
    W = torch.exp(Lc[:, :, None, :] - Lc[:, None, 1:, :])                                    # (B, T+1, T, N): [tau, sigma]
    W = W * torch.tril(torch.ones(T + 1, T, dtype=dt), -1)[None, :, :, None]                 # only the writes before
    V0 = torch.zeros((N, N + 1), dtype=dt); V0[:, N] = 1.0
    base = torch.exp(Lc)[..., None] * V0                                                     # (B, T+1, N, N+1)
    Vall, done = base.expand(B, T + 1, N, N + 1), None
    for k in range(loops):
        Vb = Vall[:, :T]                                        # the state before every sub-step
        c = torch.zeros((B, T, N, N + 1), dtype=dt)
        for o, part in enumerate(VALUE):
            if part is None:
                continue
            a = hot[:, :, part[1]]                              # (B, T, N)
            w = p[:, :, o, None, None]
            if part[0] == "set":
                x = eye1[t[:, :, part[2]]]
            elif part[0] == "clear":
                x = none.expand(B, T, N + 1)
            elif part[0] == "as":
                x = torch.einsum("btn,btnk->btk", hot[:, :, part[2]], Vb)
            elif part[0] == "lift":
                cur = torch.einsum("btn,btnk->btk", a, Vb)
                x = torch.einsum("btn,btnk->btk", cur[:, :, :N], Vb) + cur[:, :, N:] * none
            else:
                b_ = hot[:, :, part[2]]
                x = torch.einsum("btn,btnk->btk", b_, Vb)
                c = c + w * b_[..., None] * torch.einsum("btn,btnk->btk", a, Vb)[:, :, None, :]
            c = c + w * a[..., None] * x[:, :, None, :]
        Vall = base + torch.einsum("bxsn,bsnk->bxnk", W, c)
        if ref is not None and done is None and (Vall - ref).abs().max().item() < tol:
            done = k + 1
    return Vall, done


def timed(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.time(); fn(); best = min(best, time.time() - t0)
    return best


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(f"threads {torch.get_num_threads()}")
    for folder, k in [("dl/tasks_1-20_v1-2/en", 1), ("dl/tasks_1-20_v1-2/en", 2), ("dl/tasks_1-20_v1-2/en", 11), ("new/K", 1), ("new/L", 9), ("new/M2", 10), ("new/B", 2)]:
        task = load(os.path.join(L, folder), k)
        if task is None:
            print(folder, k, "skipped"); continue
        data = soft.Stories(task)
        B, S, N = data.said.shape[0], data.said.shape[1], data.N
        out = [f"{folder.split('/')[-1]}{k}: {B} stories, {S} rows, {N} things, {len(task.forms)} kinds of statements"]
        for name, sharp in (("soft, as at the start of learning", 1.0), ("random hard routing (logits x 30)", 30.0)):
            layer = soft.Layer(data, seed=1)
            with torch.no_grad():
                pv = torch.softmax(sharp * layer.value.double() + layer.maskv.double(), -1)
                pa = torch.softmax(sharp * layer.also.double() + layer.maskv.double(), -1)
                ref = sequential(data, pv, pa)
                got, done = parallel(data, pv, pa, 2 * S, ref)
                err = (got - ref).abs().max().item()
            out.append(f"  {name}: equal after {done} loops (largest difference after all loops {err:.1e})")
        # time: forward and backward, float32, soft
        layer = soft.Layer(data, seed=1)
        def seq():
            pv, _, pa = layer.parts(); h = sequential(data, pv, pa); (h[:, -1] ** 2).sum().backward()
        def par():
            pv, _, pa = layer.parts(); h, _ = parallel(data, pv, pa, 4); (h[:, -1] ** 2).sum().backward()
        a, b = timed(seq), timed(par)
        out.append(f"  forward and backward, the value memory: step by step {1000 * a:.0f} ms; in parallel with 4 loops {1000 * b:.0f} ms ({a / b:.1f} times)")
        print("\n".join(out), flush=True)


main()
