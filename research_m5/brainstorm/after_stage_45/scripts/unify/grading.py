"""Can the register be found inside the table? From CLUTRR 2-3 link stories (links given), write every story as an
equation g(answer) = sum of sign*g(link) and find the integer gradings g: relation -> Z that satisfy them (null space).
Then: how many finite states are needed besides the grading to tell the answer (partition refinement)."""
import ast, csv, sys, itertools
from collections import Counter, defaultdict
import numpy as np
D = sys.argv[1]
rows = list(csv.DictReader(open(f"{D}/gen23_train.csv")))
rels = sorted({t for r in rows for t in ast.literal_eval(r["edge_types"])} | {r["target_text"] for r in rows})
ix = {r: i for i, r in enumerate(rels)}
def path(r):
    edges, types = ast.literal_eval(r["story_edges"]), ast.literal_eval(r["edge_types"])
    a, b = ast.literal_eval(r["query_edge"])
    near = defaultdict(list)
    for (x, y), t in zip(edges, types):
        near[x].append((y, t, 1)); near[y].append((x, t, -1))
    back, q = {a: None}, [a]
    while q:
        at = q.pop(0)
        for to, t, s in near[at]:
            if to not in back:
                back[to] = (at, t, s); q.append(to)
    out = []
    while back.get(b) is not None:
        b, t, s = back[b]; out.append((t, s))
    return out[::-1]
eqs = []
for r in rows:
    p = path(r)
    v = np.zeros(len(rels)); v[ix[r["target_text"]]] += 1
    for t, s in p: v[ix[t]] -= s
    eqs.append((v, p, r["target_text"]))
A = np.array([v for v, _, _ in eqs])
# least-violated integer grading: search over small integers is too big; use SVD null space of the consistent majority
u, s, vt = np.linalg.svd(A, full_matrices=False)
print("relations", len(rels), "equations", len(A), "smallest singular values", np.round(s[-4:], 3))
g = vt[-1]; g = g / g[np.argmax(np.abs(g))]
print("weakest direction (normalised):", {r: round(float(x), 2) for r, x in zip(rels, g)})
# robust: iterate - drop equations violated by the current best integer rounding
# try scale so father=1
for scale_rel in ["father"]:
    gi = np.round(g / g[ix[scale_rel]]).astype(int)
    ok = np.abs(A @ gi) < 1e-9
    print("integer grading (father=1):", {r: int(x) for r, x in zip(rels, gi)})
    print(f"stories whose answer it satisfies: {ok.mean()*100:.2f}% of {len(A)}")
    bad = Counter((tuple(p), t) for (v, p, t), o in zip(eqs, ok) if not o)
    print("violations (most common):", bad.most_common(5))
