"""A tiny typed instruction set over exact memory, enumerated, checked exactly - on bAbI tasks one.py does not take.
Memory: every statement = (subject, kind, objects); stores are keyed overwrite cells: store[group][subject] = set of
objects (latest statement wins). Groups of kinds are found from the data: kinds whose object slots take words from
overlapping sets write one store. Programs (types: thing -> thing/set/status), depth <= 3:
  GET g x | GET g2 (GET g1 x) | GET g3 (OTHER (INV g2 (GET g1 x))) | STATUS g x a
and every program ends in SAY: a table value -> answer word learnt by counting on the stories taught on."""
import re, sys, itertools, time
from collections import Counter, defaultdict
STOP = {"the", "a", "an", "to", "in", "of", "is", "was", "did", "will", "go", "get", "what", "where", "why", "color", "or", "either", "back", "there"}
def parse(s):
    w = re.findall(r"[A-Za-z]+", s)
    subj = w[0].lower()
    objs = [w[i + 1].lower() for i in range(len(w) - 1) if w[i].lower() in ("the", "a", "an")]
    if not objs and len(w) > 2: objs = [w[-1].lower()]
    kind = " ".join("_" if (x.lower() in objs) else x.lower() for x in w[1:])
    return subj, kind, objs
def stories(path):
    out, told = [], []
    for line in open(path):
        n, rest = line.rstrip("\n").split(" ", 1)
        if n == "1": told = []
        if "\t" in rest:
            q, a = rest.split("\t")[:2]
            out.append((list(told), q.strip(), a.strip()))
        else: told.append(parse(rest))
    return out
def groups(train):
    slot = defaultdict(set)
    for told, _, _ in train:
        for s, k, o in told: slot[k] |= set(o) or {"<none>"}
    kinds, g = sorted(slot), {}
    for k in kinds:                                         # union kinds whose object words overlap
        for h in list(g):
            if len(slot[k] & slot[h]) >= 0.5 * min(len(slot[k]), len(slot[h])): g[k] = g[h]; break
        else: g[k] = k
    return g
def stores(told, g):
    st = defaultdict(dict)
    for i, (s, k, o) in enumerate(told):
        st[g.get(k, k)][s] = (frozenset(o), i)
    return st
def get(st, gr, x):
    v = st[gr].get(x)
    return None if v is None else (next(iter(v[0])) if len(v[0]) == 1 else v[0])
def run(prog, st, x, a):
    try:
        if prog[0] == "GET": return get(st, prog[1], x)
        if prog[0] == "GET2": y = get(st, prog[1], x); return None if y is None or not isinstance(y, str) else get(st, prog[2], y)
        if prog[0] == "SAME":                                 # latest other thing with the same value, then its value in another store
            y = get(st, prog[1], x)
            if not isinstance(y, str): return None
            others = sorted(((i, z) for z, (v, i) in st[prog[1]].items() if z != x and v == frozenset([y]) and z in st[prog[2]]), reverse=True)
            return get(st, prog[2], others[0][1]) if others else None
        if prog[0] == "STATUS":
            v = st[prog[1]].get(x)
            if v is None or a is None: return None
            return "only" if v[0] == frozenset([a]) else "among" if a in v[0] else "absent"
    except Exception: return None
def progs(G):
    G = sorted(set(G.values()))
    yield from (("GET", g) for g in G)
    yield from (("STATUS", g) for g in G)
    yield from (("GET2", g, h) for g in G for h in G)
    yield from (("SAME", g, h) for g in G for h in G if g != h)
def asked(q, names, objs):
    w = [x.lower() for x in re.findall(r"[A-Za-z]+", q)]
    x = next((t for t in w if t in names), None)
    a = next((t for t in reversed(w) if t in objs and t != x), None)
    qk = " ".join("<x>" if t == x else "<a>" if t == a else t for t in w)
    return x, a, qk
def learn(train, test):
    G = groups(train)
    names = {s for told, _, _ in train + test for s, _, _ in told}
    objs = {o for told, _, _ in train + test for _, _, os in told for o in os}
    by = defaultdict(list)
    for told, q, ans in train:
        x, a, qk = asked(q, names, objs); by[qk].append((stores(told, G), x, a, ans))
    chosen, tried = {}, 0
    for qk, items in by.items():
        best = None
        for p in progs(G):
            tried += 1
            say = defaultdict(Counter)
            vals = [run(p, st, x, a) for st, x, a, _ in items]
            for v, (_, _, _, ans) in zip(vals, items):
                if v is not None and not isinstance(v, frozenset): say[v][ans] += 1
            table = {v: c.most_common(1)[0][0] for v, c in say.items()}
            right = sum(table.get(v) == ans for v, (_, _, _, ans) in zip(vals, items))
            size = len(p) + len([v for v in table if table[v] != v])         # (a SAY that is not the identity costs)
            if best is None or (right, -size) > (best[0], -best[1]): best = (right, size, p, table)
        chosen[qk] = best
        print(f"   question kind '{qk}': {best[2]}  explains {best[0]}/{len(items)} taught; SAY entries not identity: {sum(1 for v in best[3] if best[3][v] != v)}")
    right = 0
    for told, q, ans in test:
        x, a, qk = asked(q, names, objs)
        if qk in chosen:
            v = run(chosen[qk][2], stores(told, G), x, a); right += chosen[qk][3].get(v) == ans
    return right / len(test), tried
B = sys.argv[1]
for t in sys.argv[2:]:
    import glob
    tr = glob.glob(f"{B}/qa{t}_*_train.txt")[0]
    train, test = stories(tr), stories(tr.replace("_train", "_test"))
    t0 = time.time()
    print(f"bAbI {t}: {len(train)} questions taught on (of {len(train)}), groups of kinds found: {len(set(groups(train).values()))}")
    for n in (len(train), 150):
        acc, tried = learn(train[:n], test)
        print(f"  taught on {n} questions -> test {100*acc:.1f}%  ({tried} programs tried, {time.time()-t0:.1f} s)")
