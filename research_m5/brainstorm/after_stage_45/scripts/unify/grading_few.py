import ast, csv, sys, random
import numpy as np
from collections import defaultdict
exec(open("grading.py").read().split("eqs = []")[0].replace('D = sys.argv[1]', 'D = sys.argv[1]'))
allrows = rows
for n in (30, 60, 100, 200):
    rng = random.Random(0); sub = rng.sample(allrows, n)
    seen = sorted({t for r in sub for t in ast.literal_eval(r["edge_types"])} | {r["target_text"] for r in sub})
    jx = {r: i for i, r in enumerate(seen)}
    A = []
    for r in sub:
        v = np.zeros(len(seen)); v[jx[r["target_text"]]] += 1
        for t, s in path(r): v[jx[t]] -= s
        A.append(v)
    A = np.array(A); u, s, vt = np.linalg.svd(A, full_matrices=True)
    null = vt[np.sum(s > 1e-9):]
    print(f"{n} stories: {len(seen)} relations met, null space dimension {len(null)}", end="; ")
    if len(null) == 1:
        g = null[0] / null[0][jx["father"]] if "father" in jx else null[0]
        print("grading:", {r: round(float(x), 2) for r, x in zip(seen, g)})
    else: print()
