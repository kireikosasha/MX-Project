#!/usr/bin/env python3
"""
CLUTRR read from the words of its stories, not from the links given beside them.

The stories of CLUTRR were written by people (who is whose kin is said in sentences of their own making, with
pronouns, two or three kin in one sentence, some of it only implied), and nearly every sentence of the test is not
met in teaching. walk.py took the links as the benchmark gives them beside the story; here the story is read:

  reading   of every two names that stand in one sentence (or in two next to each other) small networks say from the
            words what the second is to the first
  the way   from the one asked about to the other, by the links read, the surest way
  joining   by the table of walk.py, the beliefs of the readers taken into the chart

The readers are taught on the stories of two and three links, in one of two ways:

  by the links   given beside those stories (the table is counted from the same links and the answers)
  --answers      by the answers alone, the links not looked at: a reader and a soft table are taught together by a
                 gradient that goes through the chart; then the readings are made hard, the table is counted from
                 them, and what it explains is counted exactly. Several are taught, those that explain the most are
                 kept, and they vote. The way goes through the most names that stand together in sentences; the
                 one thing given beyond the answers is a hint: a word of an answer that stands in a sentence is
                 the likelier to be what the sentence says (--short --bare: the first form, without either)

In the test nothing is taken but the words of the story and the two names asked about.

    python text.py --data DIR                (DIR holds the files of CLUTRR as tables: gen23_train.csv, gen23_test.csv)
    python text.py --data DIR --answers
    python text.py --data DIR --answers --apart 0.3      (a trial without the test: sentences of three kinds in ten
                                                          are not taught on, the stories made of them are tried)
    python text.py --data DIR --encoder roberta-base --readers 1 --steps 6000     (a video card: the reader is a network
                                                                                   taught beforehand on plain text)
"""
import argparse
import ast
import csv
import heapq
import math
import os
import random
import re
import time

import torch

import walk

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORD = re.compile(r"\[[^\]]+\]|[A-Za-z]+|'s|[.,!?]")
BACK = {"father": ("son", "daughter"), "mother": ("son", "daughter"), "son": ("father", "mother"), "daughter": ("father", "mother"),
        "brother": ("brother", "sister"), "sister": ("brother", "sister"), "husband": ("husband", "wife"), "wife": ("husband", "wife"),
        "grandfather": ("grandson", "granddaughter"), "grandmother": ("grandson", "granddaughter"),
        "grandson": ("grandfather", "grandmother"), "granddaughter": ("grandfather", "grandmother"),
        "aunt": ("nephew", "niece"), "uncle": ("nephew", "niece"), "nephew": ("uncle", "aunt"), "niece": ("uncle", "aunt")}
SAID = ["nothing"] + sorted(BACK)                                # what the links beside the stories may say of a pair
WORDS = sorted(BACK) + ["daughter-in-law", "father-in-law", "mother-in-law", "son-in-law"]    # --answers: the words of the answers


def sentences(story):
    """the words of a story, sentence by sentence; a name is its number (by its first coming) -> sentences, names"""
    names, out = [], [[]]
    for w in WORD.findall(story):
        if w[0] == "[":
            if w[1:-1] not in names:
                names.append(w[1:-1])
            out[-1].append(names.index(w[1:-1]))
        else:
            out[-1].append(w.lower())
            if w in ".!?":
                out.append([])
    return [s for s in out if s], names


def near(told):
    """the pairs of names of which something may be said: both in one sentence, or in two next to each other"""
    who = [sorted({w for w in s if w.__class__ is int}) for s in told]
    out = set()
    for i, here in enumerate(who):
        for x in here:
            for y in here + (who[i + 1] if i + 1 < len(who) else []):
                if x != y:
                    out |= {(x, y), (y, x)}
    return sorted(out)


def together(told):
    """the pairs of names that stand in one sentence"""
    return {(x, y) for s in told for x in s for y in s if x.__class__ is int and y.__class__ is int and x != y}


def view(told, x, y, both=False):
    """the sentences that name x or y (both: that name both, if there are such), x written <0>, y <1>, every other
    name <o>"""
    some = [s for s in told if x in s and y in s] if both else []
    some = some or [s for s in told if x in s or y in s]
    return ["<0>" if w == x else "<1>" if w == y else "<o>" if w.__class__ is int else w for s in some for w in s]


def apart(path, share):
    """a trial of reading sentences never met, made of the stories taught on alone: such a share of the kinds of
    sentences is kept apart -> the numbers of the stories with none of them (to teach on) and of the stories made of
    them only (to try on)"""
    def kinds(story):
        got = []
        for sentence in re.split(r"(?<=[.!?])\s+", story.strip()):
            names = []

            def number(m):
                if m.group(1) not in names:
                    names.append(m.group(1))
                return f"<{names.index(m.group(1))}>"
            got.append(re.sub(r"\[([^\]]+)\]", number, sentence))
        return got
    of = [kinds(row["story"]) for row in csv.DictReader(open(path, newline="", encoding="utf-8"))]
    every = sorted({k for ks in of for k in ks})
    kept = set(random.Random(0).sample(every, int(share * len(every))))
    return ({i for i, ks in enumerate(of) if not any(k in kept for k in ks)}, {i for i, ks in enumerate(of) if all(k in kept for k in ks)})


def stories(path, only=None):
    """a file of CLUTRR (only: its stories of these numbers) -> [(sentences, the pairs near, {pair: what the second
    is to the first} by the links given beside the story, the pair asked about, the answer, how many links, what is
    added to the story: 1 nothing, 2 a second way, 3 a link aside, 4 a link apart)]"""
    out = []
    for i, row in enumerate(csv.DictReader(open(path, newline="", encoding="utf-8"))):
        if only is not None and i not in only:
            continue
        told, names = sentences(row["story"])
        if "genders" not in row:                                 # (a file with nothing but the story, the two asked about and the answer)
            asked = tuple(names.index(name) if name in names else None for name in ast.literal_eval(row["query"]))
            added = [i + 1 for i, w in enumerate(("clean", "supporting", "irrelevant", "disconnected")) if os.path.basename(path).startswith(w)]
            out.append((told, near(told), {}, asked, {"neice": "niece"}.get(row["target"], row["target"]), 3, added[0] if added else 1))
            continue
        people = [g.split(":") for g in row["genders"].split(",")]
        number = {name: names.index(name) for name, _ in people if name in names}
        edges, types, ask = (ast.literal_eval(row[x]) for x in ("story_edges", "edge_types", "query_edge"))
        gold = {}
        for (a, b), t in zip(edges, types):
            if people[a][0] in number and people[b][0] in number and t in BACK:
                x, y = number[people[a][0]], number[people[b][0]]
                gold[(x, y)] = t
                gold.setdefault((y, x), BACK[t][people[a][1] == "female"])
        asked = tuple(number.get(people[i][0]) for i in ask)
        out.append((told, near(told), gold, asked, row["target_text"], len(types), int(row["task_name"].split("_")[1].split(".")[0])))
    return out


class Reader(torch.nn.Module):
    """a small network: the words of a view -> what <1> is to <0>"""

    def __init__(self, vocab, width):
        super().__init__()
        self.vocab = {w: i + 1 for i, w in enumerate(sorted(vocab))}
        self.embed = torch.nn.Embedding(len(self.vocab) + 1, 32, padding_idx=0)
        self.gru = torch.nn.GRU(32, width, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(4 * width, len(SAID))

    def forward(self, texts):
        most = max(len(t) for t in texts)
        ids = torch.tensor([[self.vocab.get(w, 0) for w in t] + [0] * (most - len(t)) for t in texts], device=DEV)
        first, second, other = (ids == self.vocab[w] for w in ("<0>", "<1>", "<o>"))
        if self.training:                                        # (a word in seven is hidden; the places of the names never)
            ids = torch.where((torch.rand(ids.shape, device=DEV) < 0.15) & ~first & ~second & ~other, torch.zeros_like(ids), ids)
        h, _ = self.gru(self.embed(ids))
        at = [(h * m[:, :, None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1) for m in (first.float(), second.float())]
        return self.out(torch.cat(at, -1))


class Encoder(torch.nn.Module):
    """the same reader with a network taught beforehand on plain text in place of the small one (for a video card)"""

    def __init__(self, name, choices):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.add_tokens(["<0>", "<1>", "<o>"], special_tokens=True)
        self.net = AutoModel.from_pretrained(name)
        self.net.resize_token_embeddings(len(self.tok))
        self.marks = self.tok.convert_tokens_to_ids(["<0>", "<1>"])
        self.out = torch.nn.Linear(2 * self.net.config.hidden_size, choices)

    def forward(self, texts):
        enc = self.tok([" ".join(t) for t in texts], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
        h = self.net(**enc).last_hidden_state
        at = [(h * m[:, :, None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1) for m in ((enc["input_ids"] == k).float() for k in self.marks)]
        return self.out(torch.cat(at, -1))


class Hinted(torch.nn.Module):
    """a reader that is given a hint: a word that may be said of a pair and stands in the view itself is the likelier"""

    def __init__(self, net, words):
        super().__init__()
        self.net, self.words, self.much = net, words, torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, texts):
        met = torch.tensor([[w in t for w in self.words] for t in map(set, texts)], device=DEV).float()
        return self.net(texts) + self.much * met


def reader(vocab, choices, a):
    """a reader not yet taught, and how it is taught: the rate, how many lessons at a step"""
    if a.encoder:
        return Encoder(a.encoder, choices).to(DEV), 1e-3, 32
    net = Reader(vocab, a.width)
    if choices != len(SAID):
        net.out = torch.nn.Linear(4 * a.width, choices)
    return net.to(DEV), 3e-3, 256


def rates(net, rate):
    """the weights of a reader as they are taught: those taught beforehand on plain text slowly, the rest at the rate"""
    old = {id(p) for m in net.modules() if isinstance(m, Encoder) for p in m.net.parameters()}
    out = [{"params": [p for p in net.parameters() if id(p) not in old], "lr": rate, "base": rate, "weight_decay": 1e-5}]
    return out + ([{"params": [p for p in net.parameters() if id(p) in old], "lr": 2e-5, "base": 2e-5, "weight_decay": 1e-5}] if old else [])


def believing(nets):
    """readers -> a function: views -> their beliefs, averaged"""
    def believe(views):
        out = []
        with torch.no_grad():
            for i in range(0, len(views), 500):
                out.append((sum(net(views[i:i + 500]).softmax(-1) for net in nets) / len(nets)).cpu())
        return torch.cat(out) if out else torch.zeros((0, 1))
    return believe


def readers(taught, a, log, said=SAID):
    """five readers taught on the views of the stories, by the links given beside them (a pair given as None is not
    a lesson) -> a function: views -> beliefs (averaged)"""
    lessons = [(view(told, x, y), said.index(gold.get((x, y), "nothing"))) for told, pairs, gold, *_ in taught for x, y in pairs
               if gold.get((x, y), "nothing") is not None]
    vocab = {w for t, _ in lessons for w in t} | {"<0>", "<1>", "<o>"}
    gold, nets = torch.tensor([g for _, g in lessons], device=DEV), []
    for n in range(a.readers):
        torch.manual_seed(a.seed + n)
        net, rate, many = reader(vocab, len(said), a)
        opt = torch.optim.Adam(rates(net, rate))
        steps = a.steps if said is SAID else 1500
        for step in range(steps):
            for group in opt.param_groups:
                group["lr"] = group["base"] * (0.05 + 0.95 * math.cos(0.5 * math.pi * step / steps))
            some = torch.randint(0, len(lessons), (many,)).tolist()
            net.train()
            loss = torch.nn.functional.cross_entropy(net([lessons[i][0] for i in some]), gold[some])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        nets.append(net)
        log(f"reader {n + 1} of {a.readers} taught ({sum(p.numel() for p in net.parameters())} weights, {len(lessons)} lessons, loss {loss.item():.3f})")
    return believing(nets)


def surest(pairs, cost, a, b):
    """the way from a to b by the pairs near on which the least is paid -> [(x, y)] or None"""
    best, heap, came = {a: 0.0}, [(0.0, a)], {}
    while heap:
        paid, x = heapq.heappop(heap)
        if x == b:
            break
        if paid > best.get(x, 1e9):
            continue
        for p, q in pairs:
            if p == x:
                c = paid + cost(p, q)
                if c < best.get(q, 1e9):
                    best[q], came[q] = c, p
                    heapq.heappush(heap, (c, q))
    if b not in came:
        return None
    out, at = [], b
    while at != a:
        out.append((came[at], at))
        at = came[at]
    return out[::-1]


def longest(pairs, a, b, most=20000):
    """the ways from a to b by these pairs that go through the most names -> [[(x, y)]]"""
    after, found, tried = {}, [], [0]
    for p, q in sorted(pairs):
        after.setdefault(p, []).append(q)

    def go(at, been):
        tried[0] += 1
        if at == b:
            found.append(list(zip(been, been[1:])))
            return
        for q in after.get(at, []):
            if q not in been and tried[0] < most:
                go(q, been + [q])
    go(a, [a])
    return [w for w in found if w and len(w) == max(map(len, found))]


def shown(by, added, links):
    print(f"the stories of the test answered rightly from their words alone, %, by the number of links ({links}):")
    print("".join(f"{n:7d}" for n in sorted(by)) + "    all")
    print("".join(f"{100 * sum(by[n]) / len(by[n]):7.1f}" for n in sorted(by)) + f"{100 * sum(map(sum, by.values())) / sum(map(len, by.values())):7.1f}")
    if len(added) > 1:
        print("by what is added to the story: " + ", ".join(f"{('', 'nothing', 'a second way', 'a link aside', 'a link apart')[m]} {100 * sum(added[m]) / len(added[m]):.1f}"
                                                            for m in sorted(added)))


def beside(taught, tried, a, rng, log, only=None):
    """the readers taught by the links given beside the stories (only: the stories of these numbers of the file)"""
    said = sum(len(gold) for _, _, gold, *_ in taught)
    seen = sum((x, y) in gold for _, pairs, gold, *_ in taught for x, y in pairs)
    log(f"taught on {len(taught)} stories; of the links given beside them {100 * seen / said:.1f}% are between names that stand near in the words")
    believe = readers(rng.sample(taught, a.stories) if a.stories else taught, a, log)

    kinds, asks, answers = {}, {}, {}                            # the table: from the links and answers of the stories taught on
    ways = [x[:3] for i, x in enumerate(walk.kin(os.path.join(a.data, a.teach), kinds, asks, answers, True)) if x[0] is not None and (only is None or i in only)]
    table = walk.Table(ways, {i: w for w, i in kinds.items()}, answers)

    def state(word):
        return ("name", answers[word]) if word in answers else ("link", kinds[word], 1) if word in kinds else None

    for name, some in tried:
        walked(some, believe, table, state, answers, SAID, log, name + "the readers taught by the links beside the stories of two and three")


def walked(tried, believe, table, state, answers, said, log, how):
    """the stories tried are answered: every pair near is read, the surest way is found, the table joins it"""
    each = []
    for told, pairs, *_ in tried:
        bel = believe([view(told, p, q) for p, q in pairs])
        each.append({pair: bel[i] for i, pair in enumerate(pairs)})

    def answered(beliefs, how):
        read, links, by, added, no_way = 0, 0, {}, {}, 0
        for (told, pairs, gold, (x, y), answer, n, more), belief in zip(tried, beliefs):
            for pair, word in gold.items():                      # (how the links are read: counted, not used)
                links += 1
                read += pair in belief and said[int(belief[pair].argmax())] == word
            steps = None                                         # the surest way: the product of how surely something is said of every pair on it
            if x is not None and y is not None:
                steps = surest(pairs, lambda p, q: -math.log(max(1e-6, 1 - float(belief[(p, q)][0]))), x, y)
            got = None
            if steps:
                leaves = []
                for pair in steps:
                    b = belief[pair].clone()
                    b[0] = 0
                    leaf = {state(said[i]): float(b[i] / b.sum()) for i in range(1, len(said)) if b[i] / b.sum() > 0.02 and state(said[i])}
                    leaves.append(leaf)
                got = table.hard(leaves, 0)
            else:
                no_way += 1
            good = got is not None and got == answers.get(answer)
            by.setdefault(n, []).append(good)
            added.setdefault(more, []).append(good)
        log(f"the links of the stories of the test read rightly from the words: {100 * read / max(1, links):.1f}% (of {links}); no way found in {no_way} stories")
        shown(by, added, how)

    answered(each, how)
    same = {}                                                    # the same sentences with the same two names marked are read the same wherever
    for (told, pairs, *_), belief in zip(tried, each):           # they stand in the stories tried: what is believed of them is put together
        for p, q in pairs:                                       # (no answer is looked at)
            same.setdefault(tuple(view(told, p, q, True)), []).append(belief[(p, q)])
    same = {k: sum(v) / len(v) for k, v in same.items()}
    answered([{(p, q): same[tuple(view(told, p, q, True))] for p, q in pairs} for told, pairs, *_ in tried],
             how + "; the same sentences read the same in all the stories tried")


def chart(leaves, table):
    """softly: what is believed of the links of a way [stories, links, words], the table [words, words, words] ->
    what the whole way makes [stories, words]: every way of putting the brackets counts the same"""
    n = leaves.shape[1]
    span = {(i, i + 1): leaves[:, i] for i in range(n)}
    for width in range(2, n + 1):
        for i in range(n - width + 1):
            parts = [torch.einsum("bl,br,lra->ba", span[(i, m)], span[(m, i + width)], table) for m in range(i + 1, i + width)]
            span[(i, i + width)] = sum(parts) / len(parts)
    return span[(0, n)]


def mind(items, vocab, a, seed, log):
    """a reader and a soft table taught together by the answers of the stories -> the reader"""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net, rate, _ = reader(vocab, len(WORDS), a)
    net = net if a.bare else Hinted(net, WORDS).to(DEV)
    table = torch.nn.Parameter((0.01 * torch.randn(len(WORDS), len(WORDS), len(WORDS))).to(DEV))
    opt = torch.optim.Adam(rates(net, rate) + [{"params": [table], "lr": a.rate, "base": a.rate}])
    groups = {}
    for views, gold in items:
        groups.setdefault(len(views), []).append((views, gold))
    total, many = sum(len(g) for g in groups.values()), min(a.batch, 48) if a.encoder else a.batch
    for step in range(a.steps):
        for group in opt.param_groups:
            group["lr"] = group["base"] * (0.05 + 0.95 * math.cos(0.5 * math.pi * step / a.steps))
        net.train()
        loss, seen = 0, 0
        for n, group in groups.items():                          # (the ways of one length together; of every length its share)
            some = rng.sample(group, min(len(group), max(4, round(many * len(group) / total))))
            beliefs = net([v for views, _ in some for v in views]).softmax(-1).view(len(some), n, len(WORDS))
            said = chart(beliefs, table.softmax(-1))
            gold = torch.tensor([g for _, g in some], device=DEV)
            loss = loss - torch.log(said[torch.arange(len(some), device=DEV), gold] + 1e-9).sum()
            seen += len(some)
        loss = loss / seen
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [table], 5.0)
        opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            log(f"  step {step}: loss {loss.item():.3f}")
    net.eval()
    return net


def alone(taught, tried, a, log):
    """the readers taught by the answers alone"""
    answers = {w: i for i, w in enumerate(WORDS)}
    items, lengths, beside_them, where = [], {}, [], []
    for at, (told, pairs, gold, (x, y), answer, n, _) in enumerate(taught):        # the way: by the names that stand together, nothing read yet
        one, steps = together(told), None
        if x is not None and y is not None:
            steps = ([] if a.short else longest(one, x, y)[:1]) or [surest(pairs, lambda p, q: 1 if (p, q) in one else 4, x, y)]
            steps = steps[0]
        if steps and len(steps) <= 4 and answer in answers:
            items.append(([view(told, p, q, True) for p, q in steps], answers[answer]))
            beside_them.append([gold.get(pair) for pair in steps])
            where.append((at, steps))
            lengths[len(steps)] = lengths.get(len(steps), 0) + 1
    vocab = {w for views, _ in items for v in views for w in v} | {"<0>", "<1>", "<o>"}
    log(f"taught on {len(items)} stories of {len(taught)}; ways by their length: {sorted(lengths.items())}")
    flat, minds = [v for views, _ in items for v in views], []
    for n in range(a.readers):
        net = mind(items, vocab, a, a.seed + n, log)
        read = believing([net])(flat).argmax(-1).tolist()
        ways, at, good, links = [], 0, 0, 0
        for (views, gold), words in zip(items, beside_them):
            mine = read[at:at + len(views)]
            at += len(views)
            ways.append(([(k, 1) for k in mine], 0, gold))
            for k, word in zip(mine, words):
                if word:
                    links += 1
                    good += WORDS[k] == word
        table = walk.Table(ways, dict(enumerate(WORDS)), answers)          # its readings made hard, the table counted from them
        explained = sum(table.hard(s, 0) == g for s, _, g in ways) / len(ways)
        log(f"mind {n + 1}: its readings and the table counted from them explain {100 * explained:.1f}% of the stories taught on; "
            f"(not used: {100 * good / max(1, links):.1f}% of the links are read as given beside the stories)")
        minds.append((explained, net, table, read))
    best = max(e for e, *_ in minds)
    kept = sorted([x for x in minds if x[0] >= best - 0.05], key=lambda x: -x[0])
    log(f"{len(kept)} of {len(minds)} minds kept")

    for name, some in tried:
        by, added, links, good, none = {}, {}, 0, 0, 0
        for told, pairs, gold, (x, y), answer, n, more in some:
            views, votes, one = [view(told, p, q, True) for p, q in pairs], {}, together(told)
            for m, (_, net, table, _) in enumerate(kept):        # every mind kept answers by its own readings and table; they vote
                bel = believing([net])(views)
                belief = {pair: bel[i] for i, pair in enumerate(pairs)}
                if m == 0:
                    for pair, word in gold.items():
                        links += 1
                        good += pair in belief and WORDS[int(belief[pair].argmax())] == word
                steps = None                                     # the way: through the most names (--short: short), and by pairs read surely
                if x is not None and y is not None:
                    sure = {pair: -math.log(max(1e-6, float(b.max()))) for pair, b in belief.items()}
                    ways = [] if a.short else longest(one, x, y)
                    steps = min(ways, key=lambda w: sum(sure[pair] for pair in w)) if ways else surest(pairs, lambda p, q: (1 if (p, q) in one else 4) + sure[(p, q)], x, y)
                if steps:
                    got = table.hard([{table.state((i, 1)): float(belief[pair][i]) for i in range(len(WORDS)) if belief[pair][i] > 0.02} for pair in steps], 0)
                    if got is not None:
                        votes[got] = votes.get(got, 0) + 1 + 0.01 * (len(kept) - m)
            none += not votes
            ok = bool(votes) and max(votes, key=votes.get) == answers.get(answer)
            by.setdefault(n, []).append(ok)
            added.setdefault(more, []).append(ok)
        log(f"(not used: {100 * good / max(1, links):.1f}% of the links of the test are read as given beside the stories); no answer in {none} stories")
        shown(by, added, name + "the readers taught by the answers alone")
    if a.lessons:
        lessons(taught, tried, items, where, kept, answers, a, log)


def lessons(taught, tried, items, where, kept, answers, a, log):
    """what the answers have taught comes back as lessons of single links. Of every link of the ways taught on the
    minds kept say what it is (the most of them; the best, when they differ evenly); a table is counted from that,
    and the stories it explains give the lessons: the links of the way as read, and of the other names that stand
    near that nothing is said. Readers of single links are taught on these as on links given beside the stories
    (text.py without --answers), and the stories tried are answered in the same way. No link given is looked at"""
    ways, at = [], 0
    for views, gold in items:
        said = []
        for i in range(at, at + len(views)):
            votes = {}
            for m, (_, _, _, read) in enumerate(kept):
                votes[read[i]] = votes.get(read[i], 0) + 1 + 0.01 * (len(kept) - m)
            said.append(max(votes, key=votes.get))
        at += len(views)
        ways.append(([(k, 1) for k in said], 0, gold))
    table = walk.Table(ways, dict(enumerate(WORDS)), answers)
    words, again = ["nothing"] + WORDS, []
    for (steps, _, gold), (n, pairs) in zip(ways, where):
        if table.hard(steps, 0) == gold:
            told, near_pairs, _, asked, answer, links, more = taught[n]
            read = {pair: WORDS[k] for pair, (k, _) in zip(pairs, steps)}
            read.update({(q, p): None for p, q in pairs if (q, p) not in read})
            again.append((told, near_pairs, read, asked, answer, links, more))
    log(f"the minds together explain {100 * len(again) / len(ways):.1f}% of the stories taught on; these give the lessons of single links")
    believe = readers(again, a, log, words)
    for name, some in tried:
        walked(some, believe, table, lambda word: table.state((WORDS.index(word), 1)), answers, words, log,
               name + "the readers taught by the answers alone, and then by what they read themselves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the folder with gen23_train.csv and gen23_test.csv")
    ap.add_argument("--teach", default="gen23_train.csv")
    ap.add_argument("--test", default="gen23_test.csv", help="the file tried on; several with commas between are tried each by itself, with a plus between they are put together")
    ap.add_argument("--answers", action="store_true", help="the readers are taught by the answers alone")
    ap.add_argument("--bare", action="store_true", help="--answers: without the hint (a word of an answer that stands in a sentence is the likelier to be what the sentence says)")
    ap.add_argument("--lessons", action="store_true", help="--answers: what the minds read of the stories taught on is given as lessons to readers of single links, "
                    "and the stories tried are answered by those")
    ap.add_argument("--short", action="store_true", help="--answers: the way is the shortest by the names that stand together (else it goes through the most names)")
    ap.add_argument("--apart", type=float, default=0, help="a trial on the stories taught on alone: such a share of the kinds of sentences is kept apart, "
                    "taught on the stories with none of them, tried on the stories made of them only")
    ap.add_argument("--readers", type=int, default=5)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--steps", type=int, default=0, help="of teaching a reader (by default 1500 by the links, 2500 by the answers)")
    ap.add_argument("--batch", type=int, default=192, help="--answers: stories at a step")
    ap.add_argument("--rate", type=float, default=0.05, help="--answers: of the soft table")
    ap.add_argument("--encoder", default="", help="a network taught beforehand on plain text, by its name (roberta-base), in place of the small reader")
    ap.add_argument("--stories", type=int, default=0, help="by the links: the readers are taught on so many stories only (0: on all)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.steps = a.steps or (2500 if a.answers else 1500)
    torch.set_num_threads(a.threads)
    rng, t0 = random.Random(a.seed), time.time()

    def log(line):
        print(f"[{time.time() - t0:5.0f} s] {line}", flush=True)

    teach, only = os.path.join(a.data, a.teach), None
    if a.apart:                                                  # (the test is not opened)
        only, kept = apart(teach, a.apart)
        taught, tried = stories(teach, only), [("", stories(teach, kept))]
        log(f"a trial: {len(taught)} stories to teach on, {len(tried[0][1])} stories made only of sentences of the kinds kept apart")
    else:                                                        # (files with a comma between are tried each by itself; with a plus, put together)
        several = len(a.test.split(",")) > 1
        taught = stories(teach)
        tried = [(os.path.basename(names.split("+")[0]) + ": " if several else "", [story for name in names.split("+") for story in stories(os.path.join(a.data, name))])
                 for names in a.test.split(",")]
    if a.answers:
        alone(taught, tried, a, log)
    else:
        beside(taught, tried, a, rng, log, only)


if __name__ == "__main__":
    main()
