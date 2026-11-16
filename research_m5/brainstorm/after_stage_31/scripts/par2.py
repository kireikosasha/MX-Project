import sys, os, time
sys.dont_write_bytecode = True
sys.argv = [sys.argv[0]]
import importlib.util, types
src = open("par.py").read().replace("\nmain()\n", "\n")
mod = types.ModuleType("par"); exec(compile(src, "par.py", "exec"), mod.__dict__)
import torch, soft
torch.set_num_threads(4)
for folder, k in [("dl/tasks_1-20_v1-2/en", 2), ("new/B", 2)]:
    task = mod.load(os.path.join(mod.L, folder), k)
    data = soft.Stories(task)
    S = data.said.shape[1]
    layer = soft.Layer(data, seed=1)
    with torch.no_grad():
        pv = torch.softmax(layer.value.double() + layer.maskv.double(), -1); pa = torch.softmax(layer.also.double() + layer.maskv.double(), -1)
        ref = mod.sequential(data, pv, pa)
        for tol in (1e-2, 1e-3):
            _, done = mod.parallel(data, pv, pa, 2 * S, ref, tol)
            print(f"{folder.split('/')[-1]}{k}: soft start, within {tol:g} of the step-by-step state after {done} loops")
    for rep in (1, 8, 32):
        big = soft.Stories(task)
        big.said, big.things = data.said.repeat(rep, 1), data.things.repeat(rep, 1, 1)
        def seq():
            pv, _, pa = layer.parts(); h = mod.sequential(big, pv, pa); (h[:, -1] ** 2).sum().backward()
        def par():
            pv, _, pa = layer.parts(); h, _ = mod.parallel(big, pv, pa, 4); (h[:, -1] ** 2).sum().backward()
        a, b = mod.timed(seq, 2), mod.timed(par, 2)
        print(f"  {40 * rep} stories at once, 4 threads: step by step {1000 * a:.0f} ms, in parallel (4 loops) {1000 * b:.0f} ms; stories a second {40 * rep / a:.0f} and {40 * rep / b:.0f}", flush=True)
