"""Is CLUTRR kinship a cascade: two counters with one-way cancellation (bicyclic monoid: up, down),
a spouse involution and a gender flip-flop?  State of the walk from A: (u, d, s0, s1), plus gender of the end.
Word -> primitive string over P (to parent), C (to child), S (to spouse), Sib (to sibling = P then C)."""
import csv, ast, glob, sys
from collections import defaultdict, Counter

PRIM = {
    'father': 'P', 'mother': 'P', 'son': 'C', 'daughter': 'C', 'husband': 'S', 'wife': 'S',
    'brother': 'PC', 'sister': 'PC', 'grandfather': 'PP', 'grandmother': 'PP',
    'grandson': 'CC', 'granddaughter': 'CC', 'uncle': 'PPC', 'aunt': 'PPC',
    'nephew': 'PCC', 'niece': 'PCC', 'father-in-law': 'SP', 'mother-in-law': 'SP',
    'son-in-law': 'CS', 'daughter-in-law': 'CS',
}

def step(state, m):
    """state = (u, d, s0, s1): up u, then down d; s0 = went to own spouse first; s1 = ended at a spouse"""
    u, d, s0, s1 = state
    if m == 'S':
        if s1:                      # spouse of spouse = back
            return (u, d, s0, 0)
        if u == 0 and d == 0:       # own spouse (toggle)
            return (u, d, 1 - s0, 0)
        return (u, d, s0, 1)
    if m == 'C':
        if s1:                      # spouse's child = own child (no step-children in CLUTRR)
            s1 = 0
        return (u, d + 1, s0, s1)
    if m == 'P':
        if s1:                      # spouse's parent: an in-law of the relative; keep as is (rare)
            return None
        if d > 0:                   # child's parent = self or spouse: cancel one way (bicyclic)
            return (u, d - 1, s0, s1)   # (self vs spouse decided below by gender)
        if s0 == 0 and u == 0 and d == 0:
            return (u + 1, d, s0, s1)
        return (u + 1, d, s0, s1)
    raise ValueError(m)

def walk(edges, genders, names):
    st = (0, 0, 0, 0)
    for i, e in enumerate(edges):
        for m in PRIM[e]:
            nxt = step(st, m)
            if nxt is None:
                return None
            # a child's parent: if the parent reached is of the other gender than where we came down from,
            # it is that one's spouse. Approximation: when cancelling a down to (0,0), compare with A's gender.
            if m == 'P' and st[1] > 0 and nxt[0] == 0 and nxt[1] == 0:
                g_end = genders[names[i + 1]]
                g_a = genders[names[0]]
                nxt = (0, 0, nxt[2] ^ (g_end != g_a), 0)
            st = nxt
    return st

def rows(path):
    for r in csv.DictReader(open(path, newline='', encoding='utf-8')):
        edges = ast.literal_eval(r['edge_types'])
        g = dict(x.split(':') for x in r['genders'].split(','))
        q = ast.literal_eval(r['query'])
        se = ast.literal_eval(r['story_edges'])
        # node names in order: the genders list is in node order
        names = [x.split(':')[0] for x in r['genders'].split(',')]
        yield edges, g, names, q, r['target'], len(edges)

table = defaultdict(Counter)
base = sys.argv[1]
train = glob.glob(base + '/*_train.csv')[0]
skipped = 0
for edges, g, names, q, target, k in rows(train):
    if any(e not in PRIM for e in edges):
        skipped += 1; continue
    st = walk(edges, g, names)
    if st is None:
        skipped += 1; continue
    table[(st, g[q[1]])][target] += 1
conf = sum(sum(c.values()) - c.most_common(1)[0][1] for c in table.values())
tot = sum(sum(c.values()) for c in table.values())
print(f'{train}: {tot} chains, {len(table)} cascade states, conflicting labels {conf} ({100*conf/tot:.2f}%), skipped {skipped}')
name = {k: c.most_common(1)[0][0] for k, c in table.items()}
for t in sorted(glob.glob(base + '/*_test.csv')):
    right = seen = n = sk = 0
    for edges, g, names, q, target, k in rows(t):
        if any(e not in PRIM for e in edges):
            sk += 1; continue
        st = walk(edges, g, names)
        n += 1
        key = (st, g[q[1]])
        if key in name:
            seen += 1
            right += name[key] == target
    print(f'  {t.split("/")[-1]:14s} {n} chains: state seen in teaching {100*seen/max(1,n):.1f}%, answered rightly {100*right/max(1,n):.1f}% (skipped {sk})')
print('states:', len(name))
