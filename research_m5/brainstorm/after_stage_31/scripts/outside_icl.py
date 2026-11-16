"""Outside-view check: does a ~1-2M network of mind.py begin to induce a NEW task's meaning in context when the
family of dreamt tasks is as narrow as the families in the published work where in-context induction emerged
(Lake & Baroni 2023: a handful of primitives)?  The family: one kind of things, one memory (value), two kinds of
statements of two things each; what a task hides is which place of each kind of statement is the thing that gets the
value (2 bits a task), and it can be told only from the answers of the stories before.
    python outside_icl.py <names:0|2> <minutes> <seed>
Read-only on the project: imports its modules, writes nothing there."""
import sys, time, random, argparse
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import torch
torch.set_num_threads(1)
import recall, mind


def narrow(rng):
    names = {"a": ["a" + str(i) for i in range(1, rng.randint(6, 9))]}
    says = []
    for _ in range(2):
        d = rng.randrange(2)
        says.append((("a", "a"), 0, ("set", d, 1 - d), "value", 0))
    asks = [(("a",), 0, ("value", (0,)), "value", 0)]
    return names, says, asks


recall.drawn = narrow
names_, minutes, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
a = argparse.Namespace(seed=seed, least=6, stories=10, ground=True, most=640, batch=12, workers=0, lr=5e-4, minutes=minutes, out="",
                       save=1e9, every=100, width=160, layers=4, heads=4, loops=1, names=names_, varied=0.5)
print(f"names={names_} minutes={minutes} seed={seed}", flush=True)
net = mind.teach([], a, "cpu", lambda s: print(s, flush=True))
print(sum(p.numel() for p in net.parameters()), "weights", flush=True)

rng = random.Random(777)
gen = recall.dreams(rng, 14)
ks = (0, 1, 2, 4, 8)
right = {k: 0 for k in ks}; total = {k: 0 for k in ks}
sr = st = 0; n = 0; t0 = time.time()
while n < int(sys.argv[4]) if len(sys.argv) > 4 else n < 100:
    eps, _, _ = next(gen)
    if len(eps) < 12:
        continue
    n += 1
    deal = mind.dealt(eps)
    later = eps[8:12]
    for k in ks:
        r, t = mind.answers(net, eps[8 - k:8], later, "cpu", deal=deal)
        right[k] += r; total[k] += t
    m = recall.Learner(eps[:8])
    r, t, _, _ = m.test(later)
    sr += r; st += t
print("fresh tasks of the same family:", n, f"({time.time() - t0:.0f} s)")
for k in ks:
    print(f"  whole answers right with {k} stories of the task before the story asked: {100 * right[k] / max(1, total[k]):.1f}%  ({right[k]}/{total[k]})")
print(f"  the search of recall.py from 8 stories, same later stories: {100 * sr / max(1, st):.1f}%")
