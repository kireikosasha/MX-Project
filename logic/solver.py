#!/usr/bin/env python3
"""
Word problems: the network reads the text and says which expression it means. It does not compute.

The exact part (calc.py) lists every expression over the numbers of the text and computes each one; what
cannot be an answer (division by zero, a result that is not positive) is thrown away before the network
looks. The network scores what is left:

    words -> embedding (+ exact marks) -> bidirectional LSTM with the Millennium cell
    a number      = the state at its place (+ exact marks)
    an expression = composed from its numbers, bottom-up, one small layer per operation
    the score     = expression x question (+ exact facts about the computed value)

"Exact marks" are things code knows for sure and a small network would have to guess: is this word repeated
in the question, is the thing counted by this number asked about, does the question name something that
stands next to some numbers and not next to the others, is the result a whole number.

Training needs only the answer: every listed expression whose value equals the answer counts as right.
With --equations the expression written by the annotator is used instead.

Two sources of problems are made by code, every epoch anew:
    --distract   copies of training problems with one more number that cannot matter
    --world      problems of the micro-world (mathworld.py): one story, different questions
More problems with answers only (no equations) are added with --extra (data/variations.jsonl).

The network has a second head over the same reading of the text, the ledger (ledger.py): a role for every
number and a kind for the question, the answer derived by code. On the micro-world it learns the exact roles,
on real problems any reading that gives the answer counts. With both heads (--heads both, the default) the
expressions answer and the ledger explains; --heads expr / ledger trains one of them.

    python solver.py --train train.csv --test dev.csv --world 2000 --extra data/variations.jsonl --out solver.pt
    python solver.py --model solver.pt --ask "Tom had 12 apples. He ate 5. How many apples does Tom have left?"
The files: https://github.com/arkilpatel/SVAMP/tree/main/data/mawps-asdiv-a_svamp
"""
import argparse
import csv
import json
import math
import random
import re
import time
import zlib
from collections import Counter
from fractions import Fraction

import numpy as np
import torch

import calc as C
import ledger as L
import mathworld
from engine import singular
from translator import MillenniumLSTM

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LEDGER = True                # the readings of the ledger are listed for every problem (not needed with --heads expr)
BUCKETS = 96
N_TOKF, N_NUMF = 3, 12
N_FACTS = C.N_FACTS + 2
SLOT = re.compile(r"number(\d+)")
TOL = 1e-4


# ---------------------------------------------------------------- problems
def read_csv(path):
    """the files of the SVAMP repository -> problems"""
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        words = r["Question"].lower().split()
        nq = len(r["Ques"].split())
        body = r["Body"].lower().split()
        q_first = words[:nq] == r["Ques"].lower().split() and words[nq:] == body and bool(body)
        qmask = [1] * nq + [0] * (len(words) - nq) if q_first else [0] * (len(words) - nq) + [1] * nq
        out.append({"words": words, "qmask": qmask, "nums": [Fraction(x) for x in r["Numbers"].split()],
                    "answer": float(r["Answer"]), "eq": C.from_prefix(r["Equation"].split()),
                    "type": r.get("Type", ""), "eq_text": r["Equation"]})
    return out


def read_jsonl(path):
    """{"text": ..., "answer": ...} per line: numbers are found in the text"""
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        p = from_text(r["text"], r.get("answer"))
        if p:
            out.append(p)
    return out


QSTART = {"how", "what", "find", "calculate", "which", "who", "when", "where"}


def tokenize(text):
    """lower case; numbers, words, signs; 's and n't stand apart, as in the training files"""
    text = text.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"(\w)(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1 \2", text)
    return re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?|n't|'[a-z]+|[a-z]+|[^\sa-z\d]", text)


def from_text(text, answer=None):
    words, nums = [], []
    for t in tokenize(text):
        if t[0].isdigit():
            words.append(f"number{len(nums)}")
            nums.append(Fraction(t.replace(",", "")))
        else:
            words.append(t)
    if len(nums) < 2:
        return None
    start = max((i for i, w in enumerate(words) if w in QSTART), default=None)       # the question: from its last question word
    if start is None:
        start = max((i for i, w in enumerate(words[:-1]) if w in ".!"), default=-1) + 1
    qmask = [0] * start + [1] * (len(words) - start)
    return {"words": words, "qmask": qmask, "nums": nums, "answer": None if answer is None else float(answer),
            "eq": None, "type": "", "eq_text": ""}


def stem(w):
    return singular(w) if w.isalpha() else w


def build_vocab(problems, min_df=3, ids=None):
    """the words with their own vector; ids: the words of a reader that has already read plain text"""
    df = Counter(w for p in problems for w in set(p["words"]) if not SLOT.fullmatch(w))
    n = len(problems)
    own = sorted(w for w, c in df.items() if c >= min_df)
    ids = dict(ids) if ids else {w: i + 2 for i, w in enumerate(own)}     # 0 padding, 1 a number
    return {"ids": ids, "df": {w: df[w] for w in ids},
            "common": sorted(w for w, c in df.items() if c >= 0.1 * n or not w.isalpha())}


def n_vocab(vocab):
    return len(vocab["ids"]) + 2 + BUCKETS


def bucket(vocab, w):
    return len(vocab["ids"]) + 2 + zlib.crc32(w.encode()) % BUCKETS


def prepare(p, vocab, answers_only):
    """adds what the exact part knows: the expressions with their values, the marks, the right ones"""
    words, nums, k = p["words"], p["nums"], len(p["nums"])
    pos = [None] * k
    for i, w in enumerate(words):
        m = SLOT.fullmatch(w)
        if m and int(m.group(1)) < k and pos[int(m.group(1))] is None:
            pos[int(m.group(1))] = i
    if k < 2 or any(x is None for x in pos):
        return False
    common = set(vocab["common"])
    stems = [stem(w) for w in words]
    q = {s for s, m, w in zip(stems, p["qmask"], words) if m and w not in common}
    b = {s for s, m, w in zip(stems, p["qmask"], words) if not m and w not in common}
    p["pos"] = pos
    p["tokf"] = [[float(m), float(not SLOT.fullmatch(w) and w not in common and (s in b if m else s in q)), float(bool(SLOT.fullmatch(w)))]
                 for w, s, m in zip(words, stems, p["qmask"])]

    def after(i, n):                                                  # the next n meaningful words after a place
        out = []
        for j in range(i + 1, len(words)):
            if SLOT.fullmatch(words[j]) or words[j] in ".?!":
                break
            if words[j] not in common:
                out.append(stems[j])
                if len(out) == n:
                    break
        return out

    unit = [after(x, 1) for x in pos]
    near = [after(x, 3) for x in pos]
    sent = {}
    for s0, s1 in sentences(words):
        for x in pos:
            if s0 <= x < s1:
                sent[x] = (s0, s1)

    def told(x, lo, hi):                                              # how much of a stretch of the text is named in the question
        ws = [stems[j] for j in range(lo, hi) if j != x and words[j] not in common and not SLOT.fullmatch(words[j])]
        return sum(s in q for s in ws) / len(ws) if ws else 0.0

    order = sorted(range(k), key=lambda i: nums[i])
    # what the question names next to some numbers and not next to the others: by the phrase, by the sentence
    apart = []
    for ctx in ([set(x) for x in near], [{stems[j] for j in range(*sent[x]) if words[j] not in common and not SLOT.fullmatch(words[j])} for x in pos]):
        named = [c & q for c in ctx]
        some = set().union(*named) - set.intersection(*named)
        apart.append([(float(bool(n & some)), float(bool(some) and not n & some)) for n in named])
    p["numf"] = [[float(p["qmask"][pos[i]]),
                  float(bool(unit[i]) and unit[i][0] in q),                                       # what it counts is asked about
                  sum(s in q for s in near[i]) / 3.0,
                  float(bool(unit[i]) and any(unit[j] == unit[i] for j in range(k) if j != i)),   # another number counts the same
                  float(order[-1] == i), float(order[0] == i),
                  told(pos[i], sent[pos[i]][0], pos[i]),                                         # who it is about is asked about
                  told(pos[i], *sent[pos[i]]),
                  *apart[0][i], *apart[1][i]] for i in range(k)]
    exprs = C.expressions(k, 2 if k <= 5 else 1)
    keep, vals = [], []
    for i, t in enumerate(exprs):
        v = C.value(t, nums)
        if v is not None and v > 0:
            keep.append(i)
            vals.append(v)
    if not keep:
        return False
    p["cand"] = [exprs[i] for i in keep]
    p["vals"] = [float(v) for v in vals]
    p["facts"] = [C.facts(exprs[i], nums, v) for i, v in zip(keep, vals)]
    for f, t in zip(p["facts"], p["cand"]):                           # do the numbers count the same thing
        ls = C.leaves(t)
        if len(ls) == 2:
            f += [float(bool(unit[ls[0]]) and unit[ls[0]] == unit[ls[1]]), 0.0]
        else:
            inner = C.leaves(t[1] if t[1][0] != "n" else t[2])
            c = [x for x in ls if x not in inner][0]
            f += [float(bool(unit[inner[0]]) and unit[inner[0]] == unit[inner[1]]),
                  float(bool(unit[c]) and unit[c] in (unit[inner[0]], unit[inner[1]]))]
    p["n_one"] = sum(1 for t in p["cand"] if C.n_ops(t) == 1)
    one, two = [], []
    for t in p["cand"]:
        if C.n_ops(t) == 1:
            one.append((C.OPS.index(t[0]), t[1][1], t[2][1]))
        else:
            side = 0 if t[1][0] != "n" else 1
            inner, c = (t[1], t[2][1]) if side == 0 else (t[2], t[1][1])
            two.append((C.OPS.index(t[0]), side, C.OPS.index(inner[0]), inner[1][1], inner[2][1], c))
    p["one"], p["two"] = one, two
    if p["answer"] is None:
        p["gold"] = []
    elif answers_only:
        p["gold"] = [j for j, v in enumerate(p["vals"]) if abs(v - p["answer"]) < TOL]
    else:
        g = C.find(p["eq"], k, 2 if k <= 5 else 1)
        p["gold"] = [keep.index(g)] if g is not None and g in keep else []
    # the ledger: every well formed reading (roles of the numbers, kind of the question) with its answer
    if k <= 7 and USE_LEDGER:
        T, kind = L.readings(k)
        out, ok = L.compute(k, [float(x) for x in nums])
        T, kind, out = T[ok], kind[ok], out[ok]
    else:
        T, kind, out = np.zeros((0, k), dtype=np.int8), np.zeros(0, dtype=np.int8), np.zeros(0)
    p["l_roles"], p["l_kind"], p["l_vals"] = torch.from_numpy(T.astype(np.int64)), torch.from_numpy(kind.astype(np.int64)), out.tolist()
    if "roles" in p:                                                  # the micro-world tells the reading itself
        hit = np.nonzero((T == np.array(p["roles"], dtype=np.int8)).all(1) & (kind == p["query"]))[0]
    elif p["answer"] is None:
        hit = []
    else:
        hit = np.nonzero(np.abs(out - p["answer"]) < TOL)[0]
    p["l_gold"] = [int(x) for x in hit]
    return True


# ---------------------------------------------------------------- numbers that do not matter
STOP = {".", ",", "?", "!", ";", ":", "and", "but", "then", "if", "so", "while", "when"}


def sentences(words):
    """[(start, end)]: a sentence ends with its full stop"""
    out, s0 = [], 0
    for i, w in enumerate(words):
        if w in ".?!":
            out.append((s0, i + 1))
            s0 = i + 1
    if s0 < len(words):
        out.append((s0, len(words)))
    return out


def unit_words(problems, vocab):
    """the words around numbers in the training texts. After a number: what is counted, when, of what kind.
    Before it in the sentence: who or what the sentence is about."""
    common, after, before = set(vocab["common"]), Counter(), Counter()
    for p in problems:
        words = p["words"]
        for x in p["pos"]:
            for w in words[x + 1:x + 4]:
                if SLOT.fullmatch(w) or w in STOP:
                    break
                if w not in common and w.isalpha():
                    after[w] += 1
        for s0, s1 in sentences(words):
            slots = [i for i in range(s0, s1) if SLOT.fullmatch(words[i])]
            if len(slots) == 1:
                before.update(w for w in words[s0:slots[0]] if w not in common and w.isalpha())

    def table(c):
        words = sorted(w for w, n in c.items() if n >= 2)
        return words, [c[w] ** 0.5 for w in words]

    return {"after": table(after), "before": table(before)}


def distract(p, rng, units, vocab):
    """a copy of the problem with one more number that cannot matter. The question names something ("on
    monday", "does mike have"); a phrase or a sentence with a number is repeated with another number and with
    that word replaced by another one:
        "14 kids on monday" -> "14 kids on monday and 31 kids on friday"
        "mike has 87 cards ." -> "paul has 67 cards . mike has 87 cards ."
    The question is still about monday, about mike: the answer is the same."""
    words, common = p["words"], set(vocab["common"])
    q = {stem(w) for w, m in zip(words, p["qmask"]) if m and w not in common}
    have = {stem(w) for w in words}

    def named(i):
        return words[i].isalpha() and words[i] not in common and stem(words[i]) in q

    places = []
    for x in p["pos"]:
        if p["qmask"][x]:
            continue
        end = x + 1
        while end < len(words) and end - x <= 5 and words[end] not in STOP and not SLOT.fullmatch(words[end]):
            end += 1
        places += [("after", x, end, j) for j in range(x + 1, end) if named(j)]
    for s0, s1 in sentences(words):
        slots = [i for i in range(s0, s1) if SLOT.fullmatch(words[i])]
        if len(slots) == 1 and words[s1 - 1] == "." and not any(p["qmask"][s0:s1]):
            places += [("before", s0, s1, j) for j in range(s0, slots[0]) if named(j)]
    if not places:
        return None
    kind, x, end, j = rng.choice(places)
    for _ in range(8):
        other = rng.choices(*units[kind])[0]
        if stem(other) not in have:
            break
    else:
        return None
    pool = [int(v) for v in p["nums"] if v.denominator == 1] or [10]
    new = Fraction(max(1, rng.choice(pool) + rng.randint(-9, 30)) if rng.random() < 0.7 else rng.randint(2, 99))
    if new in p["nums"]:
        return None
    k = len(p["nums"])
    if kind == "before":                                       # a sentence about someone else, put before
        copy = [other if i == j else (f"number{k}" if SLOT.fullmatch(words[i]) else words[i]) for i in range(x, end)]
        w2 = words[:x] + copy + words[x:]
        m2 = p["qmask"][:x] + [0] * len(copy) + p["qmask"][x:]
    else:
        copy = [f"number{k}"] + [other if i == j else words[i] for i in range(x + 1, end)]
        if rng.random() < 0.5:
            w2 = words[:end] + ["and"] + copy + words[end:]
            m2 = p["qmask"][:end] + [0] * (len(copy) + 1) + p["qmask"][end:]
        else:
            w2 = words[:x] + copy + ["and"] + words[x:]
            m2 = p["qmask"][:x] + [0] * (len(copy) + 1) + p["qmask"][x:]
    # the numbers are named in the order of the text
    seen, nums2 = {}, []
    for i, w in enumerate(w2):
        m = SLOT.fullmatch(w)
        if m:
            old = int(m.group(1))
            if old not in seen:
                seen[old] = len(nums2)
                nums2.append(new if old == k else p["nums"][old])
            w2[i] = f"number{seen[old]}"
    out = {"words": w2, "qmask": m2, "nums": nums2, "answer": p["answer"], "eq": None, "type": p["type"], "eq_text": ""}
    if not prepare(out, vocab, True) or not out["gold"]:
        return None
    if any(seen[k] in C.leaves(out["cand"][g]) for g in out["gold"]):       # the new number must not give the answer
        return None
    out["l_gold"] = [g for g in out["l_gold"] if int(out["l_roles"][g, seen[k]]) == 0]
    return out


# ---------------------------------------------------------------- the network
class Solver(torch.nn.Module):
    def __init__(self, n_words, dim=96, hid=128, d=128, drop=0.4, marks=True, facts=True, ledger=False, pairs=0, layers=1):
        super().__init__()
        self.cfg = {"n_words": n_words, "dim": dim, "hid": hid, "d": d, "drop": drop, "marks": marks, "facts": facts,
                    "ledger": ledger, "pairs": pairs, "layers": layers}
        self.marks, self.use_facts, self.ledger, self.pairs = marks, facts, ledger, pairs
        if ledger:                                  # the second head: the role of every number, the kind of the question
            self.role = torch.nn.Sequential(torch.nn.Linear(3 * d, d), torch.nn.Tanh(), torch.nn.Linear(d, len(L.ROLES)))
            self.kind = torch.nn.Linear(d, len(L.QUERIES))
            if pairs:                               # how well the roles of two numbers go together, and a role with the kind
                self.pair = torch.nn.Linear(3 * d, len(L.ROLES) * pairs)
                self.kind_pair = torch.nn.Linear(d, len(L.QUERIES) * pairs)
        self.emb = torch.nn.Embedding(n_words, dim, padding_idx=0)
        self.tokf = torch.nn.Linear(N_TOKF, dim, bias=False)
        self.fw, self.bw = MillenniumLSTM(dim, hid), MillenniumLSTM(dim, hid)
        self.more = torch.nn.ModuleList([torch.nn.ModuleList([MillenniumLSTM(2 * hid, hid), MillenniumLSTM(2 * hid, hid)])
                                         for _ in range(layers - 1)])
        self.num = torch.nn.Linear(2 * hid + N_NUMF, d)
        self.att = torch.nn.Linear(2 * hid, 1)
        self.q = torch.nn.Linear(4 * hid, d)
        self.comp = torch.nn.ModuleList([torch.nn.Linear(2 * d, d) for _ in C.OPS])
        self.s_node, self.s_mix = torch.nn.Linear(d, d), torch.nn.Linear(d, d, bias=False)
        self.s_fact = torch.nn.Linear(N_FACTS, d, bias=False)
        self.out = torch.nn.Linear(d, 1)
        self.drop = torch.nn.Dropout(drop)

    def compose(self, op, left, right):
        z = torch.cat([left, right], -1)
        out = z.new_zeros(z.shape[0], self.cfg["d"])
        for o in range(len(C.OPS)):
            m = op == o
            if m.any():
                out[m] = torch.tanh(self.comp[o](z[m]))
        return out

    def read(self, ids, tokf=None):
        """the reader: word ids (B, T) -> the state at every word (B, T, 2 hid), and which places are words"""
        mask = (ids != 0).float()
        x = self.emb(ids)
        if tokf is not None and self.marks:
            x = x + self.tokf(tokf)
        x = self.drop(x)
        h = torch.cat([self.fw(x, mask), self.bw(x, mask, reverse=True)], -1)
        for fw, bw in self.more:                                # further layers read the states of the layer below
            z = self.drop(h)
            h = h + torch.cat([fw(z, mask), bw(z, mask, reverse=True)], -1)
        return h, mask

    def forward(self, batch):
        """-> for every problem the scores of its expressions and the scores of its readings in the ledger"""
        B, T = len(batch), max(len(p["ids"]) for p in batch)
        ids = torch.zeros(B, T, dtype=torch.long)
        tokf = torch.zeros(B, T, N_TOKF)
        qm = torch.zeros(B, T)
        for i, p in enumerate(batch):
            n = len(p["ids"])
            ids[i, :n] = torch.tensor(p["ids"])
            tokf[i, :n] = torch.tensor(p["tokf"])
            qm[i, :n] = torch.tensor(p["qmask"], dtype=torch.float)
        ids, tokf, qm = ids.to(DEV), tokf.to(DEV), qm.to(DEV)
        h, mask = self.read(ids, tokf)
        h = self.drop(h)
        a = torch.softmax(self.att(h).squeeze(-1).masked_fill(mask == 0, -1e30), -1)
        whole = (a.unsqueeze(-1) * h).sum(1)
        qn = (qm.unsqueeze(-1) * h).sum(1) / qm.sum(1, keepdim=True).clamp(min=1)                 # the question part
        quest = torch.tanh(self.q(torch.cat([whole, qn], -1)))                                    # (B, d)
        # numbers
        rows, cols, numf, off = [], [], [], [0]
        for i, p in enumerate(batch):
            rows += [i] * len(p["pos"])
            cols += p["pos"]
            numf += p["numf"]
            off.append(off[-1] + len(p["pos"]))
        numf = torch.tensor(numf, device=DEV)
        if not self.marks:
            numf = torch.zeros_like(numf)
        N = torch.tanh(self.num(torch.cat([h[rows, cols], numf], -1)))                            # (all numbers, d)
        # expressions with one operation, then with two
        one = torch.tensor([(o, a + off[i], b + off[i]) for i, p in enumerate(batch) for o, a, b in p["one"]],
                           dtype=torch.long, device=DEV).reshape(-1, 3)
        two = torch.tensor([(o2, s, o1, a + off[i], b + off[i], c + off[i]) for i, p in enumerate(batch) for o2, s, o1, a, b, c in p["two"]],
                           dtype=torch.long, device=DEV).reshape(-1, 6)
        facts = torch.tensor([f for p in batch for f in p["facts"]], device=DEV) if self.use_facts else None
        node1 = self.compose(one[:, 0], N[one[:, 1]], N[one[:, 2]])
        if len(two):
            inner = self.compose(two[:, 2], N[two[:, 3]], N[two[:, 4]])
            left = torch.where(two[:, 1:2] == 0, inner, N[two[:, 5]])
            right = torch.where(two[:, 1:2] == 0, N[two[:, 5]], inner)
            node2 = self.compose(two[:, 0], left, right)
        scores, i1, i2 = [], 0, 0
        for i, p in enumerate(batch):
            n1, n2 = len(p["one"]), len(p["two"])
            node = node1[i1:i1 + n1] if n2 == 0 else torch.cat([node1[i1:i1 + n1], node2[i2:i2 + n2]], 0)
            node = self.drop(node)
            z = self.s_node(node) + self.s_mix(node * quest[i])
            if self.use_facts:
                z = z + self.s_fact(facts[i1 + i2:i1 + i2 + n1 + n2])
            i1, i2 = i1 + n1, i2 + n2
            scores.append(self.out(torch.tanh(z)).squeeze(-1))
        reads = [None] * B
        if self.ledger:
            kind_lp = torch.log_softmax(self.kind(self.drop(quest)), -1)
            for i, p in enumerate(batch):
                n = N[off[i]:off[i + 1]]
                k = n.shape[0]
                other = (n.sum(0, keepdim=True) - n) / max(1, k - 1)                      # what the other numbers are
                feats = torch.cat([n, quest[i].expand(k, -1), other], -1)
                T, kinds = p["l_roles"].to(DEV), p["l_kind"].to(DEV)
                reads[i] = torch.log_softmax(self.role(feats), -1).gather(1, T.t()).sum(0) + kind_lp[i, kinds]
                if self.pairs and len(T):
                    e = self.pair(feats).view(k, len(L.ROLES), self.pairs)
                    e = e[torch.arange(k, device=DEV)[None, :], T] * (T != 0).unsqueeze(-1)   # (readings, k, pairs)
                    total = e.sum(1)
                    ke = self.kind_pair(quest[i]).view(len(L.QUERIES), self.pairs)[kinds]
                    reads[i] = reads[i] + 0.5 * (total.pow(2).sum(-1) - e.pow(2).sum((1, 2))) + (ke * total).sum(-1)
        return scores, reads


def encode(problems, vocab, rng=None, word_drop=0.0, rare=40):
    """word ids; in training a rare word is sometimes replaced by its anonymous bucket"""
    ids, df = vocab["ids"], vocab["df"]
    for p in problems:
        row = []
        for w in p["words"]:
            if SLOT.fullmatch(w):
                row.append(1)
            elif w in ids and not (rng and df[w] < rare and rng.random() < word_drop):
                row.append(ids[w])
            else:
                row.append(bucket(vocab, w))
        p["ids"] = row


def by_value(vals, scores):
    """the scores of expressions or readings -> the probability of every answer"""
    out = {}
    for v, x in zip(vals, torch.softmax(scores, 0).tolist()):
        out[round(v, 6)] = out.get(round(v, 6), 0.0) + x
    return out


def best(d):
    return max(d, key=d.get) if d else None


def answer(model, problems, bs=64):
    """-> for every problem the answers: by the expressions, by the ledger, by both; and the two distributions"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(problems), bs):
            part = problems[i:i + bs]
            scores, reads = model(part)
            for p, s, r in zip(part, scores, reads):
                e = by_value(p["vals"], s)
                led = by_value(p["l_vals"], r) if r is not None and len(r) else {}
                both = {v: math.log(e.get(v, 0.0) + 1e-4) + math.log(led.get(v, 0.0) + 1e-4) for v in set(e) | set(led)} if led else e
                out.append({"expr": best(e), "ledger": best(led), "both": best(both), "e": e, "l": led,
                            "j": int(s.argmax()), "r": int(r.argmax()) if led else None})
    return out


def right(p, v):
    return v is not None and abs(v - p["answer"]) < TOL


def accuracy(model, problems, total=None):
    """-> the share of right answers by the expressions, by the ledger, by both; and the answers"""
    got = answer(model, problems)
    n = total or len(problems)
    return {m: sum(right(p, g[m]) for p, g in zip(problems, got)) / n for m in ("expr", "ledger", "both")}, got


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Solver(**ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.to(DEV), ck["vocab"]


def measure(path, test_path):
    """a trained model on a file of problems"""
    model, vocab = load(path)
    test = read_csv(test_path)
    n = len(test)
    test = [p for p in test if prepare(p, vocab, True)]
    encode(test, vocab)
    acc, got = accuracy(model, test, n)
    print(f"{n} problems: expressions {100 * acc['expr']:.1f}" + (f", ledger {100 * acc['ledger']:.1f}" if model.ledger else ""))


def ask(path, text):
    model, vocab = load(path)
    ck = {"vocab": vocab}
    p = from_text(text)
    if not p or not prepare(p, ck["vocab"], True):
        print("the text needs at least two numbers")
        return
    encode([p], ck["vocab"])
    g = answer(model, [p])[0]
    with torch.no_grad():
        s, r = model([p])
    print(f"answer: {g['expr']:g}" + ("" if g["ledger"] is None else "   (the ledger agrees)" if abs(g["ledger"] - g["expr"]) < 1e-6 else
                                      f"   (the ledger reads it differently: {g['ledger']:g})"))
    s = torch.softmax(s[0], 0)
    for j in s.argsort(descending=True)[:2].tolist():
        print(f"  expression  {C.show(p['cand'][j], p['nums'])} = {p['vals'][j]:g}   ({100 * float(s[j]):.0f} %)")
    if r[0] is not None and len(r[0]):
        r = torch.softmax(r[0], 0)
        for j in r.argsort(descending=True)[:2].tolist():
            print(f"  ledger      {L.show(p['l_roles'][j].tolist(), int(p['l_kind'][j]), p['nums'])} = {p['l_vals'][j]:g}   ({100 * float(r[j]):.0f} %)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="")
    ap.add_argument("--test", default="")
    ap.add_argument("--model", default="", help="a trained model: with --ask it answers, with --test it is measured")
    ap.add_argument("--ask", default="", help="a problem as text: prints the expressions the model prefers")
    ap.add_argument("--extra", default="", help="more problems: jsonl with text and answer (answers only)")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--drop", type=float, default=0.4)
    ap.add_argument("--word-drop", type=float, default=0.3)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--equations", action="store_true", help="learn from the annotated equations, not from the answers")
    ap.add_argument("--no-marks", action="store_true")
    ap.add_argument("--no-facts", action="store_true")
    ap.add_argument("--heads", default="both", choices=["both", "expr", "ledger"],
                    help="what is trained: the expressions, the ledger, or both (then the expressions answer and the ledger explains)")
    ap.add_argument("--pairs", type=int, default=16, help="the ledger also scores pairs of roles (the size of the pair vectors, 0: no)")
    ap.add_argument("--distract", type=float, default=0.5, help="share of problems that get a copy with a number that does not matter, every epoch")
    ap.add_argument("--world", type=int, default=0, help="problems from the micro-world (mathworld.py) added every epoch")
    ap.add_argument("--layers", type=int, default=1, help="layers of the reader")
    ap.add_argument("--pretrained", default="", help="a reader that has read plain text (pretrain.py): its words, its size, its weights")
    ap.add_argument("--sets", type=int, default=0, help="make the problems of code for so many epochs once and go round them (0: new every epoch)")
    ap.add_argument("--limit", type=int, default=0, help="use only so many of the training problems (how much is learnt from how little)")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    if a.ask:
        return ask(a.model, a.ask)
    if a.model:
        return measure(a.model, a.test)
    global USE_LEDGER
    USE_LEDGER = a.heads != "expr"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    train_all, test = read_csv(a.train), read_csv(a.test)
    reader = torch.load(a.pretrained, map_location="cpu", weights_only=False) if a.pretrained else None
    if reader:
        a.dim, a.hid, a.layers = reader["cfg"]["dim"], reader["cfg"]["hid"], reader["cfg"]["layers"]
    vocab = build_vocab(train_all, a.min_df, reader["ids"] if reader else None)
    extra = read_jsonl(a.extra) if a.extra else []
    n_test = len(test)
    train_all = [p for p in train_all if prepare(p, vocab, not a.equations)]
    extra = [p for p in extra if prepare(p, vocab, True)]
    test = [p for p in test if prepare(p, vocab, True)]
    reach = sum(1 for p in test if p["gold"])
    reach_l = sum(1 for p in test if p["l_gold"])
    usable = [p for p in train_all if p["gold"] or p["l_gold"]]
    extra = [p for p in extra if p["gold"] or p["l_gold"]]
    rng.shuffle(usable)
    n_val = len(usable) // 10
    val, train = usable[:n_val], usable[n_val:] + extra
    if a.limit:                                              # the words are then also only those of what is used
        train = usable[n_val:][:a.limit] + extra
        seen = [mathworld.problem(random.Random(a.seed + 7)) for _ in range(2000)] if a.world else []
        vocab = build_vocab(train + seen, a.min_df, reader["ids"] if reader else None)
        for p in train:
            prepare(p, vocab, not a.equations or p["eq"] is None)
        for p in val + test:
            prepare(p, vocab, True)
    print(f"train {len(train)} (of them generated {len(extra)}), held out {len(val)}, test {len(test)} of {n_test}; "
          f"the right value exists among the expressions for {reach} test problems, among the readings of the ledger for {reach_l}; "
          f"words {n_vocab(vocab)}", flush=True)
    model = Solver(reader["cfg"]["n_words"] if reader else n_vocab(vocab), a.dim, a.hid, a.d, a.drop, not a.no_marks, not a.no_facts,
                   a.heads != "expr", a.pairs, a.layers)
    if reader:                                                   # the reader starts from what it has read
        left = model.load_state_dict(reader["state"], strict=False)
        assert not left.unexpected_keys, left.unexpected_keys
    model.to(DEV)
    mode = "ledger" if a.heads == "ledger" else "expr"            # with both heads the expressions answer, the ledger explains
    print("parameters:", sum(x.numel() for x in model.parameters()), "on", DEV, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    encode(val + test, vocab)
    base, units = train, unit_words(train, vocab)

    def made():
        """the problems made by code for one epoch: copies with a number that does not matter, the micro-world"""
        out = []
        for p in base:
            if rng.random() < a.distract:
                p2 = distract(p, rng, units, vocab)
                if p2 and rng.random() < 0.3:                # sometimes two such numbers
                    p2 = distract(p2, rng, units, vocab) or p2
                if p2:
                    out.append(p2)
        for _ in range(a.world):
            p2 = mathworld.problem(rng)
            if prepare(p2, vocab, True) and (p2["gold"] or p2["l_gold"]):
                out.append(p2)
        return out

    sets = [made() for _ in range(a.sets)]
    chosen = (-1.0, 0, None)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        train = list(base) + (sets[ep % a.sets] if a.sets else made())
        encode(train, vocab, rng, a.word_drop)
        order = list(range(len(train)))
        rng.shuffle(order)
        chunks = [sorted(order[i:i + a.batch * 20], key=lambda j: len(train[j]["words"])) for i in range(0, len(order), a.batch * 20)]
        batches = [c[i:i + a.batch] for c in chunks for i in range(0, len(c), a.batch)]
        rng.shuffle(batches)
        tot = 0.0
        for bi, bt in enumerate(batches):
            at = (ep + bi / len(batches)) / a.epochs                  # a short warm-up, then a cosine down
            for gr in opt.param_groups:
                gr["lr"] = a.lr * (at / 0.05 if at < 0.05 else 0.5 * (1 + math.cos(math.pi * (at - 0.05) / 0.95)))
            part = [train[j] for j in bt]
            loss = 0.0
            scores, reads = model(part)
            for p, s, r in zip(part, scores, reads):
                if p["gold"] and a.heads != "ledger":
                    loss = loss + torch.logsumexp(s, 0) - torch.logsumexp(s[p["gold"]], 0)
                if p["l_gold"] and r is not None:
                    loss = loss + torch.logsumexp(r, 0) - torch.logsumexp(r[p["l_gold"]], 0)
            loss = loss / len(part)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if (ep + 1) % a.eval_every and ep + 1 < a.epochs:
            continue
        va, _ = accuracy(model, val)
        te, _ = accuracy(model, test, n_test)
        if va[mode] >= chosen[0]:
            chosen = (va[mode], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"epoch {ep + 1}: loss {tot / len(batches):.3f}  held out {100 * va[mode]:.1f}  test: expressions {100 * te['expr']:.1f}, "
              f"ledger {100 * te['ledger']:.1f}, both {100 * te['both']:.1f}  ({time.time() - t0:.0f} s)", flush=True)
    model.load_state_dict(chosen[2])
    te, got = accuracy(model, test, n_test)
    print(f"chosen by the held out part: epoch {chosen[1] + 1}, held out {100 * chosen[0]:.1f}, test {100 * te[mode]:.1f} "
          f"(expressions {100 * te['expr']:.1f}, ledger {100 * te['ledger']:.1f}, both {100 * te['both']:.1f})")
    by = {}
    for p, g in zip(test, got):
        for key in (p["type"], f"{len(p['nums'])} numbers"):
            r = by.setdefault(key, [0, 0])
            r[0] += right(p, g[mode])
            r[1] += 1
    print("  ".join(f"{k}: {100 * r / n:.1f} ({n})" for k, (r, n) in sorted(by.items())))
    ok = [right(p, g[mode]) for p, g in zip(test, got)]
    if len(ok) > 200:                                        # mistakes were read only among the first 200
        print(f"the first 200: {100 * sum(ok[:200]) / 200:.1f}; the rest ({len(ok) - 200}): {100 * sum(ok[200:]) / (len(ok) - 200):.1f}")
    shown = 0
    for p, g in zip(test, got):
        if shown < a.show and not right(p, g[mode]):
            shown += 1
            led = L.show(p["l_roles"][g["r"]].tolist(), int(p["l_kind"][g["r"]]), p["nums"]) if g["r"] is not None else ""
            print("   ", " ".join(p["words"]), "|", [float(x) for x in p["nums"]], "| chose", g[mode], "| expression", C.show(p["cand"][g["j"]], p["nums"]),
                  "| ledger", led, "| right", p["answer"], p["eq_text"])
    if a.out:
        torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "cfg": model.cfg, "vocab": vocab}, a.out)


if __name__ == "__main__":
    main()
