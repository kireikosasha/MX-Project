"""Diagnostic for the reader lens: do dependency paths between two names carry the kin relation, and do path
signatures of the test recur in training far more often than whole views do? Counting only, no weights learnt."""
import ast, csv, re, sys, collections
sys.path.insert(0, sys.argv[1])
import spacy
nlp = spacy.load("en_core_web_sm")
BACK = {"father": ("son", "daughter"), "mother": ("son", "daughter"), "son": ("father", "mother"), "daughter": ("father", "mother"),
        "brother": ("brother", "sister"), "sister": ("brother", "sister"), "husband": ("husband", "wife"), "wife": ("husband", "wife"),
        "grandfather": ("grandson", "granddaughter"), "grandmother": ("grandson", "granddaughter"),
        "grandson": ("grandfather", "grandmother"), "granddaughter": ("grandfather", "grandmother"),
        "aunt": ("nephew", "niece"), "uncle": ("nephew", "niece"), "nephew": ("uncle", "aunt"), "niece": ("uncle", "aunt")}
KIN = set(BACK) | {"dad", "mom", "mum", "grandma", "grandpa", "child", "children", "parent", "kid", "sibling", "spouse", "boy", "girl", "granddad"}

def load(path):
    out = []
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        raw = row["story"]
        names, text, spans = [], "", []
        pos = 0
        for m in re.finditer(r"\[([^\]]+)\]", raw):
            text += raw[pos:m.start()]
            spans.append((len(text), len(text) + len(m.group(1)), m.group(1)))
            text += m.group(1)
            pos = m.end()
        text += raw[pos:]
        people = [g.split(":") for g in row["genders"].split(",")]
        edges, types = ast.literal_eval(row["story_edges"]), ast.literal_eval(row["edge_types"])
        gold = {}
        for (a, b), t in zip(edges, types):
            if t in BACK:
                x, y = people[a][0], people[b][0]
                gold[(x, y)] = t
                gold.setdefault((y, x), BACK[t][people[a][1] == "female"])
        out.append((text, spans, gold))
    return out

def sig(doc, i, j):
    """the shortest path in the tree from token i to token j: lemmas of the words between and the arcs"""
    def up(k):
        chain = [k]
        while doc[chain[-1]].head.i != chain[-1]:
            chain.append(doc[chain[-1]].head.i)
        return chain
    a, b = up(i), up(j)
    common = next((k for k in a if k in b), None)
    if common is None:
        return None
    left = a[:a.index(common) + 1]
    right = b[:b.index(common)][::-1]
    path = left + right
    arcs = [("^" + doc[k].dep_) for k in left[:-1]] + [("v" + doc[k].dep_) for k in right]
    words = [doc[k].lemma_.lower() for k in path[1:-1]]
    # possessive / kin words hanging on the path (her son, X's son)
    hang = sorted({c.lemma_.lower() for k in path for c in doc[k].children if c.dep_ in ("poss", "case") and c.i not in path})
    return tuple(words), tuple(arcs), tuple(hang)

def links(data):
    out = []
    for doc, (text, spans, gold) in zip(nlp.pipe([d[0] for d in data], batch_size=64), data):
        tok = {}
        for s, e, name in spans:
            span = doc.char_span(s, e, alignment_mode="expand")
            if span is not None:
                tok.setdefault(name, []).append(span.root.i)
        for (x, y), t in gold.items():
            if x not in tok or y not in tok:
                continue
            best = None
            for i in tok[x]:
                for j in tok[y]:
                    if doc[i].sent.start == doc[j].sent.start:
                        s = sig(doc, i, j)
                        if s and (best is None or len(s[1]) < len(best[1])):
                            best = s
            out.append(((x, y), t, best))
    return out

train, test = load(sys.argv[2]), load(sys.argv[3])
L, T = links(train), links(test)
same = lambda L: [l for l in L if l[2] is not None]
for name, ls in (("teach", L), ("test", T)):
    s = same(ls)
    kin_on = sum(1 for _, t, (w, a, h) in s if any(k in w for k in KIN))
    gold_on = sum(1 for _, t, (w, a, h) in s if t in w or any(t2 in w for t2 in BACK.get(t, ())) or any(BACK[t][0] in w or BACK[t][1] in w for _ in [0]))
    print(f"{name}: links {len(ls)}, in one sentence {len(s)} ({100*len(s)/len(ls):.1f}%); a kin word on the path {100*kin_on/len(s):.1f}%")
# lexicon of full signatures -> majority kin, from the links of teaching
full = collections.defaultdict(collections.Counter)
mid = collections.defaultdict(collections.Counter)    # backoff: the words on the path and the arcs, without hanging words
low = collections.defaultdict(collections.Counter)    # backoff: the kin words on the path and the first and last arc
def key_low(w, a):
    return (tuple(k for k in w if k in KIN), a[:1], a[-1:])
for _, t, s in same(L):
    full[s][t] += 1; mid[s[:2]][t] += 1; low[key_low(*s[:2])][t] += 1
st = same(T)
hit = collections.Counter(); right = collections.Counter()
for _, t, s in st:
    for lvl, table, k in (("full", full, s), ("mid", mid, s[:2]), ("low", low, key_low(*s[:2]))):
        if k in table:
            hit[lvl] += 1
            right[lvl] += table[k].most_common(1)[0][0] == t
            break
    else:
        hit["none"] += 1
n = len(st)
print(f"test links in one sentence: {n}")
for lvl in ("full", "mid", "low", "none"):
    print(f"  answered by {lvl:5s}: {100*hit[lvl]/n:5.1f}% of links, right {100*right[lvl]/max(1,hit[lvl]):5.1f}%")
print(f"  overall right among one-sentence links (no weights): {100*sum(right.values())/n:.1f}%")
print(f"  distinct full signatures in teaching {len(full)}, mid {len(mid)}, low {len(low)}")

NEUTRAL = {"father": "parent", "mother": "parent", "son": "child", "daughter": "child", "brother": "sibling", "sister": "sibling",
           "husband": "spouse", "wife": "spouse", "grandfather": "grandparent", "grandmother": "grandparent", "grandson": "grandchild",
           "granddaughter": "grandchild", "aunt": "auncle", "uncle": "auncle", "nephew": "nibling", "niece": "nibling"}
fullN = collections.defaultdict(collections.Counter); lowN = collections.defaultdict(collections.Counter); midN = collections.defaultdict(collections.Counter)
for _, t, s in same(L):
    fullN[s][NEUTRAL[t]] += 1; midN[s[:2]][NEUTRAL[t]] += 1; lowN[key_low(*s[:2])][NEUTRAL[t]] += 1
hit = collections.Counter(); right = collections.Counter()
for _, t, s in st:
    for lvl, table, k in (("full", fullN, s), ("mid", midN, s[:2]), ("low", lowN, key_low(*s[:2]))):
        if k in table:
            hit[lvl] += 1; right[lvl] += table[k].most_common(1)[0][0] == NEUTRAL[t]; break
    else:
        hit["none"] += 1
print("gender taken apart (the kind of kin only):")
for lvl in ("full", "mid", "low", "none"):
    print(f"  answered by {lvl:5s}: {100*hit[lvl]/n:5.1f}% of links, right {100*right[lvl]/max(1,hit[lvl]):5.1f}%")
print(f"  overall right among one-sentence links: {100*sum(right.values())/n:.1f}%")
# how pure are the full signatures in teaching (an upper bound on any reader that sees only the path)
pure = sum(c.most_common(1)[0][1] for c in fullN.values()) / sum(sum(c.values()) for c in fullN.values())
pureg = sum(c.most_common(1)[0][1] for c in full.values()) / sum(sum(c.values()) for c in full.values())
print(f"  purity of full signatures in teaching: kind of kin {100*pure:.1f}%, kin with sex {100*pureg:.1f}%")

INV = {"parent": "child", "child": "parent", "sibling": "sibling", "spouse": "spouse", "grandparent": "grandchild",
       "grandchild": "grandparent", "auncle": "nibling", "nibling": "auncle"}
def one_kin(s):
    w, a, h = s
    ks = [k for k in list(w) + list(h) if k in NEUTRAL]
    return ks[0] if len(set(ks)) == 1 else None
def label(t, k):
    if NEUTRAL[t] == NEUTRAL[k]:
        return "same"
    if NEUTRAL[t] == INV[NEUTRAL[k]]:
        return "inverse"
    return "other"
def where(s, k):
    w, a, h = s
    return (a, "on" if k in w else "hang")
dirs = collections.defaultdict(collections.Counter)
dirs2 = collections.defaultdict(collections.Counter)
for _, t, s in same(L):
    k = one_kin(s)
    if k:
        dirs[where(s, k)][label(t, k)] += 1
        dirs2[(s[1][:1], s[1][-1:], "on" if k in s[0] else "hang")][label(t, k)] += 1
tot = ok = cov = 0
for _, t, s in st:
    k = one_kin(s)
    if not k:
        continue
    tot += 1
    c = dirs.get(where(s, k)) or dirs2.get((s[1][:1], s[1][-1:], "on" if k in s[0] else "hang"))
    if c:
        cov += 1
        guess = c.most_common(1)[0][0]
        ok += guess == label(t, k) and guess != "other"
print(f"one kin word on the path or hanging on it: {tot} of {n} test links ({100*tot/n:.1f}%)")
print(f"  meaning = that word or its inverse, the side chosen by the arcs alone (shared by all kin words): covered {100*cov/tot:.1f}%, right {100*ok/tot:.1f}% of them all")
print(f"  distinct arc patterns in teaching: {len(dirs)} (backoff {len(dirs2)})")
print("  labels in teaching:", sum((c for c in dirs.values()), collections.Counter()))
pur = sum(c.most_common(1)[0][1] for c in dirs.values()) / sum(sum(c.values()) for c in dirs.values())
print(f"  purity of the side by arcs in teaching (resubstitution): {100*pur:.1f}%")
# split teaching in half: learn arcs->side on one half, try on the other (same distribution, no test noise)
import random
S = same(L); random.Random(0).shuffle(S); A, B = S[:len(S)//2], S[len(S)//2:]
d = collections.defaultdict(collections.Counter)
for _, t, s in A:
    k = one_kin(s)
    if k: d[where(s, k)][label(t, k)] += 1
tot = ok = 0
for _, t, s in B:
    k = one_kin(s)
    if not k: continue
    tot += 1
    c = d.get(where(s, k))
    ok += bool(c) and c.most_common(1)[0][0] == label(t, k)
print(f"  teaching split in halves: side by arcs right {100*ok/tot:.1f}% on the other half")
