#!/usr/bin/env python3
"""
Reading free language with the ground.

The learner (acquire.py) learns a task from 20 stories in template language. Its forms then give, for every template
sentence, a reading: the scheme and which things play which roles. Here a small network learns from those readings to
choose a reading for a sentence in ANY wording. Two things keep it right where it has never seen the words:

  sense    the ground notes what happens in a story as it is read (a value changed, less than nothing left, a chain came
           back to its start, a question got no answer). What of this never happens in the stories the learner read
           itself is strange; of the readings of a new story the reader takes those with the least strange in them.
           No answers are needed for this.
  answers  on free stories with answers every reading of a sentence with which the answers of its story come out right
           is a lesson; an answer seldom leaves one reading, and which of them is meant the network works out over
           many stories. Nobody labels the free sentences.

    python reader.py --data tasks_1-20_v1-2/en --k 1 --free free/stories --name b1 [--vectors glove50.gz]

What is given to the reader beyond the learner: nothing about the task. The names of things are those the learner met
in argument places; a candidate reading is one of the learner's schemes with those names in its roles; a role with few
values (a direction, a time) may be named by any word. With --vectors every word also brings a vector made elsewhere
from plain text (GloVe): a dictionary. --school lets the reader first learn the lessons other tasks left (--lessons).
"""
import argparse
import glob
import gzip
import itertools
import json
import os
import random
import re
import sys

import torch
import torch.nn as nn

import acquire as A
import core as G

ERR = (IndexError, KeyError, ValueError, TypeError)
CLOSED = ("D", "T")     # roles with few values (directions, times): a word the learner never met may name one of them


def load_vectors(path, limit=50000):
    """vectors of words made elsewhere from much plain text (GloVe): the reader's dictionary -> {word: row}, rows"""
    index, rows = {}, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            part = line.rstrip().split(" ")
            if len(part) > 10:
                if not rows:
                    rows.append([0.0] * (len(part) - 1))
                index[part[0]] = len(rows)
                rows.append([float(x) for x in part[1:]])
                if len(rows) > limit:
                    break
    return index, torch.tensor(rows)


def words_of(text):
    return tuple(re.findall(r"[^\W_]+|[.,?]", text.lower()))


def load_free(folder, name):
    """the paraphrased stories of a task: {"train": [...], "test": [...]}; a story is [(template words, free words, answer)]"""
    out = {"train": {}, "test": {}}
    for f in sorted(glob.glob(os.path.join(folder, name + "_*.json"))):
        for st in json.load(open(f, encoding="utf-8"))["stories"]:
            out[st["part"]][st["id"]] = [(words_of(x["s"]), words_of(x.get("p") or x["s"]),
                                          [g.strip() for g in x["a"].lower().split(",")] if x.get("a") else None) for x in st["sentences"]]
    return {p: [v for _, v in sorted(d.items())] for p, d in out.items()}


class Reader:
    def __init__(self, mind, template, vectors=None):
        self.m, norm = mind, mind.lex.norm
        self.names, self.kinds = set(), {}                       # the things met alone in an argument place, in the form
        self.shapes, self.sorts = {}, {}                         # the learner's lexicon gives them, and their sorts; the
        self.closed, self.turn = {}, {}                          # schemes with their roles; which sorts stand in a role;
        parses = []                                              # the values of closed roles; the schemes met turned round
        for story in template:
            for words, gold in story:
                p = mind.parse(words, gold is not None)
                if p and p[0] != "NOP":
                    parses.append((gold is not None, p))
                    for r, vals in p[2].items():
                        if r not in ("O", "N") + CLOSED and len(vals) == 1:
                            self.names.add(norm(vals[0]))
                            self.kinds.setdefault(norm(vals[0]), set()).add(mind.sort_of(vals[0].split()))
        for quest, p in parses:
            a = self.roles(p[2])
            if any(not v for v in a.values()):
                continue
            key = (quest, p[0], p[1] if not isinstance(p[1], list) else tuple(p[1]))
            for o in p[2].get("O", ()):
                if o in mind.lex.flip:
                    self.turn.setdefault(key, o)
            roles = tuple(sorted((r, len(v)) for r, v in a.items()))
            self.shapes.setdefault(key, {}).setdefault(roles, 0)
            self.shapes[key][roles] += 1
            for r, vals in a.items():
                if r in CLOSED:
                    self.closed.setdefault((key, r), set()).update(vals)
                elif r != "N":
                    self.sorts.setdefault((key, r), set()).update(*(self.kinds[v] for v in vals))
        self.longest = max([len(n.split()) for n in self.names] + [1])
        self.vindex, self.vectors = vectors if vectors else ({}, None)
        self.vocab, self.net, self.nets = {"<pad>": 0, "<unk>": 1}, None, []
        self.start = None                                        # a network that has read other tasks: every new one starts from it
        self.say = dict(mind.said_all, **{s: c.most_common(1)[0][0] for s, c in mind.said_n.items()})
        self.usual = set()                                       # what happens in this world: the notes of the ground on
        for story in template:                                   # the stories as the learner itself reads them
            picks = []
            for words, gold in story:
                p = mind.parse(words, gold is not None)
                picks.append((p[0], p[1], dict(self.roles(p[2]), O=p[2].get("O", [])), None) if p and p[0] != "NOP" else (None, 0, {}, None))
            self.usual.update(self.answers(story, picks)[1])

    def roles(self, a):
        """the roles of a reading of the learner, as the reader keeps them: things by their names, no words of the form"""
        norm = self.m.lex.norm
        return {r: [v if r in ("N",) + CLOSED else norm(v) for v in vals if r in ("N",) + CLOSED or norm(v) in self.names]
                for r, vals in a.items() if r != "O"}

    # ---------------------------------------------------------------- candidates
    def mentions(self, words):
        """the things and the numbers named in a sentence: [(start, end, is it a number, the name)]"""
        out, i, norm = [], 0, self.m.lex.norm
        while i < len(words):
            for n in range(min(self.longest, len(words) - i), 0, -1):
                name = norm(" ".join(words[i:i + n]))
                if name in self.names and (n == 1 or len(name.split()) == n):
                    out.append((i, i + n, False, name))
                    i += n
                    break
            else:
                if words[i].isdigit():
                    out.append((i, i + 1, True, words[i]))
                i += 1
        return out

    def candidates(self, words, quest):
        """every reading the learner's schemes allow for this sentence: [(scheme or None, number, roles, tokens)]"""
        ment = self.mentions(words)

        def tokens(scheme, at, more=()):
            toks, pos = [f"<{scheme}>", "<?>" if quest else "<.>"] + list(more), 0
            for j, (s, e, num, _) in enumerate(ment):
                toks += list(words[pos:s]) + [f"<{at[j]}>" if j in at else ("<#>" if num else "<E>")]
                pos = e
            return toks + list(words[pos:])
        out = [(None, 0, {}, tokens("NOP", {}))]
        for (q, kind, k), shapes in self.shapes.items():
            if q != quest:
                continue
            for roles in sorted(shapes, key=lambda r: -shapes[r])[:2]:
                slots = [r for r, n in roles for _ in range(n) if r not in CLOSED]
                given = [r for r, n in roles for _ in range(n) if r in CLOSED]
                if len(slots) > len(ment):
                    continue
                for pick in itertools.permutations(range(len(ment)), len(slots)):
                    if any((slots[i] == "N") != ment[j][2] for i, j in enumerate(pick)):
                        continue
                    if any(slots[i] != "N" and not self.kinds[ment[j][3]] & self.sorts[((q, kind, k), slots[i])] for i, j in enumerate(pick)):
                        continue                                 # a thing of a sort the learner never met in this role
                    if any(slots[i] == slots[i2] and pick[i] > pick[i2] for i in range(len(slots)) for i2 in range(i + 1, len(slots))):
                        continue                                 # the same role twice: one order is enough
                    at = {j: slots[i] for i, j in enumerate(pick)}
                    for vals in itertools.product(*(sorted(self.closed[((q, kind, k), r)]) for r in given)):
                        a = {}
                        for i, j in enumerate(pick):
                            a.setdefault(slots[i], []).append(ment[j][3])
                        for r, v in zip(given, vals):            # a closed role: every value it had is offered, whatever
                            a.setdefault(r, []).append(v)        # the words; the network tells which the sentence means
                        more = [f"<{r}={v}>" for r, v in zip(given, vals)]
                        out.append((kind, list(k) if isinstance(k, tuple) else k, a, tokens(f"{kind}:{k}", at, more)))
                        if (q, kind, k) in self.turn:            # the same, turned round
                            out.append((kind, out[-1][1], dict(a, O=[self.turn[(q, kind, k)]]), tokens(f"{kind}:{k}~", at, more)))
        return out

    # ---------------------------------------------------------------- the ground
    def answers(self, story, picks):
        """the readings chosen for the sentences of a story -> how many of its answers are right, and what was notable
        in the story read so: the notes of the ground, and the questions left without an answer"""
        w, right, notes = G.World(self.m.lex), 0, []
        for (words, gold), (kind, k, a, _) in zip(story, picks):
            k = tuple(k) if isinstance(k, list) else k
            try:
                if gold is None:
                    if kind is not None:
                        w.tell(kind, k, a)
                else:
                    got = w.ask(kind, k, a) if kind is not None else None
                    if got is None:
                        notes.append((kind, k, "no answer"))
                    right += self.m.right(got, gold, self.say)
            except ERR:
                notes.append((kind, k, "fails"))
        return right, w.notes + notes

    def grade(self, story, cands, pick):
        """-> right answers, and minus how much of what happens never happened in the stories the learner read itself"""
        right, notes = self.answers(story, [c[i] for c, i in zip(cands, pick)])
        return right, -sum(n not in self.usual for n in notes)

    def allowed(self, story, cands, pick):
        """for every sentence of a story read with all answers right: all its readings with which they stay right (and
        nothing gets stranger). An answer seldom leaves one reading: which of them is meant, many stories tell"""
        best, out = self.grade(story, cands, pick), []
        for s in range(len(story)):
            out.append([alt for alt in range(len(cands[s]))
                        if alt == pick[s] or self.grade(story, cands, pick[:s] + [alt] + pick[s + 1:]) >= best])
        return out

    def climb(self, story, cands, logp, start, blind):
        """from the given readings, by the best change of one sentence at a time, to the readings that fit best:
        most right answers (not looked at when blind), least of what never happened in the stories the learner read
        itself, likeliest"""
        def score(pick):
            right, odd = self.grade(story, cands, pick)
            return (0 if blind else right, odd, sum(lp[i] for lp, i in zip(logp, pick)))
        pick, best = list(start), score(start)
        for _ in range(2 * len(story)):
            step = max(((score(pick[:s] + [alt] + pick[s + 1:]), s, alt) for s in range(len(story))
                        for alt in range(len(cands[s])) if alt != pick[s]), default=None)
            if step is None or step[0] <= best:
                break
            best, pick[step[1]] = step[0], step[2]
        return best, pick

    def solve(self, story, cands, logp, tries=3, blind=False):
        """the readings that fit the story best, sought from what the network thinks and from chance starts"""
        rng, top = random.Random(0), [max(range(len(lp)), key=lambda i: lp[i]) for lp in logp]
        best = max((self.climb(story, cands, logp, top if t == 0 else [rng.randrange(len(c)) for c in cands], blind)
                    for t in range(1 if blind else tries)), key=lambda x: x[0])
        return best[0][0], best[1]

    # ---------------------------------------------------------------- the network
    def encode(self, toks, grow=False):
        """every token has a place of its own in the network once it is met in a lesson; a word the dictionary has
        brings its vector too - for a word never met in a lesson that is all that is known of it"""
        out = []
        for t in toks:
            if grow:
                self.vocab.setdefault(t, len(self.vocab))
            out.append((self.vocab.get(t, 1), self.vindex.get(t, 0)))
        return out

    def scores(self, cand_lists):
        """log-probabilities of the candidates of every sentence"""
        flat = [self.encode(c[3]) for cands in cand_lists for c in cands]
        n = max(len(x) for x in flat)
        x = torch.tensor([[t for t, _ in row] + [0] * (n - len(row)) for row in flat])
        v = torch.tensor([[t for _, t in row] + [0] * (n - len(row)) for row in flat])
        out, at = [], 0
        every = [net(x, v) for net in ([self.net] if self.net.training else self.nets)]
        for cands in cand_lists:                                 # in reading, the last networks answer together
            out.append(sum(torch.log_softmax(s[at:at + len(cands)], 0) for s in every) / len(every))
            at += len(cands)
        return out

    def fit(self, lessons, epochs=30, seed=0, keep=3):
        """lessons: [(candidates, the index of the right one - or of all that may be right)]"""
        torch.manual_seed(seed)
        random.seed(seed)
        for cands, _ in lessons:
            for c in cands:
                self.encode(c[3], grow=True)
        self.net = Net(len(self.vocab), self.vectors)
        if self.start is not None:                               # what was learnt of the language on other tasks is kept
            mine = self.net.state_dict()
            for name, old in self.start.items():
                if name == "emb.weight":
                    mine[name][:old.shape[0]] = old
                elif name != "vec.weight":
                    mine[name] = old.clone()
            self.net.load_state_dict(mine)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        for _ in range(epochs):
            random.shuffle(lessons)
            for i in range(0, len(lessons), 32):
                part = lessons[i:i + 32]
                loss = -sum(torch.logsumexp(lp[t], 0) if isinstance(t, list) else lp[t]
                            for lp, (_, t) in zip(self.scores([c for c, _ in part]), part)) / len(part)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        self.nets = (self.nets + [self.net])[-keep:]

    def school(self, files, epochs=12):
        """before the task: the lessons other tasks left (their sentences with the readings found for them) are learnt
        once; the tokens of the ground - schemes, roles - mean the same in every task, and so do the words"""
        lessons = []
        for f in files:
            lessons += [([(None, 0, {}, toks) for toks in cands], t) for cands, t in json.load(open(f, encoding="utf-8"))]
        self.fit(lessons, epochs=epochs, seed=7)
        self.start = {k: v.clone() for k, v in self.net.state_dict().items()}
        self.nets = []
        return len(lessons)

    def read(self, stories, sense=True):
        """reading without the answers: the network's choice for every sentence, then (with sense) the changes after
        which less in the story makes no sense -> right answers, and the readings"""
        right, picks = 0, []
        with torch.no_grad():
            for story in stories:
                cands = [self.candidates(w, g is not None) for w, g in story]
                lp = [p.tolist() for p in self.scores(cands)]
                pick = [max(range(len(p)), key=lambda i: p[i]) for p in lp]
                if sense:
                    pick = self.solve(story, cands, lp, blind=True)[1]
                right += self.answers(story, [c[i] for c, i in zip(cands, pick)])[0]
                picks += [(c, i) for c, i in zip(cands, pick)]
        return right, picks


class Net(nn.Module):
    def __init__(self, n_words, vectors=None, width=48):
        super().__init__()
        self.emb = nn.Embedding(n_words, width, padding_idx=0)
        self.vec = nn.Embedding.from_pretrained(vectors if vectors is not None else torch.zeros(1, 1), freeze=True)
        self.lift = nn.Linear(self.vec.embedding_dim, width)
        self.rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x, v):
        known = torch.where(x < self.emb.num_embeddings, x, torch.ones_like(x))      # a token met after this network was made
        e = self.emb(known) + self.lift(self.vec(v)) * (v > 0).unsqueeze(-1)
        h, _ = self.rnn(e)
        h = h.masked_fill((x == 0).unsqueeze(-1), -1e9).max(1).values
        return self.out(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the folder with the template task files")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--free", required=True, help="the folder with the paraphrased stories")
    ap.add_argument("--name", required=True, help="the name of the task in the paraphrase files")
    ap.add_argument("--free-train", type=int, default=40, help="free stories with answers to learn from")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--alone", type=int, default=0, help="rounds of reading the test stories without their answers")
    ap.add_argument("--vectors", default="", help="a GloVe file (.gz) with vectors of words: the reader's dictionary")
    ap.add_argument("--school", default="", help="lesson files of other tasks (a glob) to learn the language from first")
    ap.add_argument("--lessons", default="", help="where to leave this task's lessons for other tasks")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    torch.set_num_threads(1)                                     # the network is tiny: one thread is enough
    template = A.read(glob.glob(os.path.join(a.data, f"qa{a.k}_*train.txt"))[0], 20)
    mind = A.Mind()
    got, total = mind.learn(template, 150)
    free = load_free(a.free, a.name)
    train = [[(p, g) for _, p, g in st] for st in free["train"][:a.free_train]]
    test = [[(p, g) for _, p, g in st] for st in free["test"]]
    same = [[(s, g) for s, _, g in st] for st in free["test"]]                # the same test stories in template language
    r = Reader(mind, template, load_vectors(a.vectors) if a.vectors else None)
    n_test = sum(g is not None for st in test for _, g in st)
    res = {"task": a.name, "learner_template_train": [got, total], "test_questions": n_test, "usual": sorted(map(str, r.usual))}

    def learner(stories):                                        # the learner alone, by its forms
        return sum(mind.right(ans, gold, r.say) for st in stories for ans, gold in mind.run(st))
    res["learner_on_template_test"] = learner(same)
    res["learner_on_free_test"] = learner(test)
    # the lessons from the template stories: the learner's own readings
    lessons = []
    for story in template:
        for words, gold in story:
            p = mind.parse(words, gold is not None)
            cands = r.candidates(words, gold is not None)
            want = None
            if p and p[0] != "NOP":
                want = (p[0], p[1], {k: sorted(v) for k, v in r.roles(p[2]).items()})
                if any(o in mind.lex.flip for o in p[2].get("O", ())):
                    want[2]["O"] = [r.turn[(gold is not None, p[0], tuple(p[1]) if isinstance(p[1], list) else p[1])]]
            hit = [i for i, c in enumerate(cands) if (c[0] is None and want is None) or
                   (want and c[0] == want[0] and c[1] == want[1] and {k: sorted(v) for k, v in c[2].items()} == want[2])]
            if hit:
                lessons.append((cands, hit[0]))
    res["template_lessons"] = len(lessons)
    if a.school:
        res["school_lessons"] = r.school(sorted(f for f in glob.glob(a.school) if os.path.basename(f) != a.name + ".json"))
    r.fit(list(lessons))
    res["reader_template_only"] = [r.read(test, False)[0], r.read(test)[0]]        # the network alone; with sense
    # the ceiling: the best the candidates allow on the test, found with the answers
    with torch.no_grad():
        ceil = 0
        for story in test:
            cands = [r.candidates(w, g is not None) for w, g in story]
            ceil += r.solve(story, cands, [lp.tolist() for lp in r.scores(cands)])[0]
    res["ceiling_with_answers"] = ceil
    print(json.dumps(res), flush=True)
    # free stories with answers only: the ground checks, the reader learns
    for rnd in range(a.rounds):
        new, solved = [], 0
        with torch.no_grad():
            for story in train:
                cands = [r.candidates(w, g is not None) for w, g in story]
                right, pick = r.solve(story, cands, [lp.tolist() for lp in r.scores(cands)])
                if right == sum(g is not None for _, g in story):
                    solved += 1
                    new += list(zip(cands, r.allowed(story, cands, pick)))
        r.fit(list(lessons) + new, seed=rnd + 1)
        alone, got = r.read(test, False)[0], r.read(test)[0]
        print(f"round {rnd + 1}: {solved}/{len(train)} free stories read with all answers right; free test: the network alone "
              f"{100 * alone / n_test:.1f}%, with sense {100 * got / n_test:.1f}%", flush=True)
        res.setdefault("rounds", []).append([solved, alone, got])
    res["reader_with_free"] = res["rounds"][-1][1:]
    if a.lessons:
        json.dump([([c[3] for c in cands], t) for cands, t in list(lessons) + new], open(a.lessons, "w", encoding="utf-8"))
    # reading on its own: the test stories WITHOUT their answers, read so that they make sense, are lessons too
    for rnd in range(a.alone):
        own = r.read(test)[1]
        r.fit(list(lessons) + new + own, seed=100 + rnd)
        alone, got = r.read(test, False)[0], r.read(test)[0]
        print(f"alone {rnd + 1}: after reading the test stories without answers: the network alone {100 * alone / n_test:.1f}%, "
              f"with sense {100 * got / n_test:.1f}%", flush=True)
        res.setdefault("alone", []).append([alone, got])
    if a.alone:
        res["reader_after_reading_alone"] = res["alone"][-1]
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
