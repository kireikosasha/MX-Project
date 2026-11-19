#!/usr/bin/env python3
"""Answer-signature reading + a solver-found machine for StepGame (stage 49 follow-up).

1. The sentence kinds (symbols = (view, sign)) are grouped by how they behave, not by their words: the signature of a
   symbol is the answers of the teaching stories where it is the only link (a minimal pair with the empty context);
   symbols with the same most frequent answer form one group. Prior used: a one-link story says what its one link
   means. Minimal pairs in one-link contexts (stories that differ in exactly one symbol) check the groups and place the
   symbols that never stand alone (and the ones whose own answers are split).
2. Every teaching story is rewritten in group symbols; stories that become the same are merged (answers counted).
3. The machine and its shape are found by the SAT solver of cells.py (cells.learn: shapes in order of their length in
   bits, asking where rival machines part) on the clean group-level stories (told at least twice, always the same
   answer). Nothing is written by hand: no shape is fixed.
4. The reading is corrected against the machine found (--correct 2, default): every symbol is read as the machine
   symbol that answers rightly the most teaching stories it stands in (exact check, two passes; the machine is kept).
   --correct 1 instead lets the solver correct it: cells.fit with lexicon over the groups (each group one raw
   symbol) and the weak symbols, taught greedily (stories no machine explains are set aside) - slower, within --budget.
   --signature pairs groups without the one-link prior: minimal pairs in two-link stories only.
5. Measured by the protocol: teaching stories (different ones) and 1000 test stories per k=1..10 (random.Random(1)).
"""
import argparse, os, pickle, random, sys, time
from collections import Counter, defaultdict

sys.argv, ARGV = ['x'], sys.argv
sys.path.insert(0, '../..')
import cells
import one

DATA = 'StepGame/Dataset/TrainVersion'
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f} s]", *a, flush=True)


def seq_of(story):
    steps = one.steps_of(story)
    return None if not steps else tuple(steps) + (("asked", story[6]),)


class Signature:
    """the same face as one.Found: built from the teaching stories, answers a story (or None)"""

    def __init__(self, taught, a, log=log):
        self.a = a
        said = {}
        for s in taught:
            seq = seq_of(s)
            if seq:
                said.setdefault(seq, Counter())[s[4]] += 1
        self.said = said
        self.every = {seq: c.most_common(1)[0][0] for seq, c in said.items()}
        self.labels = sorted({y for y in self.every.values()})
        log(f"{len(taught)} teaching stories, {len(said)} different, {len({x for s in said for x in s})} symbols")
        self.group(log)
        self.machine(log)
        if a.correct == 1:
            self.correct(log)
        elif a.correct == 2:
            self.check(log)
        self.explains = sum(self.run(s) == y for s, y in self.every.items()) / len(self.every)
        self.explains_all = sum(c[self.run(s)] for s, c in self.said.items()) / len(taught)
        log(f"the final machine answers {100 * self.explains:.2f}% of the {len(self.every)} different teaching stories "
            f"rightly ({100 * self.explains_all:.2f}% of all {len(taught)} teaching stories)")

    # ------------------------------------------------------------ 1. groups by answer signature
    def group(self, log):
        a = self.a
        if a.signature == "pairs":
            return self.group_pairs(log)
        alone = defaultdict(Counter)                     # symbol -> answers of the stories where it is the only link
        for seq, c in self.said.items():
            if len(seq) == 2:
                alone[seq[0]].update(c)
        sig = {}
        for s, c in alone.items():
            y, n = c.most_common(1)[0]
            sig[s] = (y, n / sum(c.values()), sum(c.values()))
        keys = Counter(y for y, share, n in sig.values() if share >= a.sure)
        name = {y: f"G{i}" for i, (y, _) in enumerate(keys.most_common())}   # (named by size, not by the answer)
        self.read = {s: name[y] for s, (y, share, n) in sig.items() if share >= a.sure}
        self.groups = sorted(name.values(), key=lambda g: int(g[1:]))
        self.asked = sorted({seq[-1] for seq in self.said})
        views = {x for seq in self.said for x in seq[:-1]}
        weak = [s for s in views if s not in self.read]
        log(f"answer signatures: {len(sig)} symbols stand alone in some story; {len(self.read)} grouped into "
            f"{len(self.groups)} groups (sizes {' '.join(str(keys[y]) for y in name)}); {len(weak)} not grouped "
            f"({sum(1 for s in weak if s in sig)} with split answers, {sum(1 for s in weak if s not in sig)} never alone)")
        # minimal pairs: stories that differ in one symbol, everything else read the same
        self.check_pairs(log)
        placed = self.place(weak, log)
        self.read.update(placed)
        self.weak = [s for s in weak]
        log(f"placed by minimal pairs: {len(placed)} of {len(weak)}")

    def group_pairs(self, log):
        """groups without the one-link prior: only minimal pairs in two-link stories (a symbol and one other link).
        Two symbols agree in a context when the stories with either of them there have the same answer; a symbol
        joins the group it agrees with most (at least `sure` of the shared contexts, at least `evidence` of them),
        else it begins a group of its own. Symbols in order of how often they are met."""
        a = self.a
        bucket = defaultdict(list)
        met = Counter()
        for seq, c in self.said.items():
            met.update(seq[:-1])
            if len(seq) == 3:
                y = self.every[seq]
                bucket[(0, seq[1])].append((seq[0], y))
                bucket[(1, seq[0])].append((seq[1], y))
        agree = defaultdict(lambda: [0, 0])
        for xs in bucket.values():
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    (p, yp), (q, yq) = xs[i], xs[j]
                    if p != q:
                        agree[(p, q)][yp != yq] += 1
                        agree[(q, p)][yp != yq] += 1
        near = defaultdict(dict)
        for (p, q), v in agree.items():
            near[p][q] = v
        groups, read = [], {}
        for s in sorted(met, key=lambda s: (-met[s], repr(s))):
            best = None
            for gi, members in enumerate(groups):
                ag = dis = 0
                for t in members:
                    v = near[s].get(t)
                    if v:
                        ag, dis = ag + v[0], dis + v[1]
                if ag + dis >= a.evidence and ag / (ag + dis) >= a.sure and (best is None or ag / (ag + dis) > best[0]):
                    best = (ag / (ag + dis), gi)
            if best is None:
                if sum(sum(v) for v in near[s].values()) >= a.evidence:   # enough contexts to begin a group
                    groups.append([s])
                    read[s] = len(groups) - 1
            else:
                groups[best[1]].append(s)
                read[s] = best[1]
        order = sorted(range(len(groups)), key=lambda g: -len(groups[g]))
        big = [g for g in order if len(groups[g]) >= a.least]
        name = {g: f"G{i}" for i, g in enumerate(big)}
        self.read = {s: name[g] for s, g in read.items() if g in name}
        self.groups = [name[g] for g in big]
        views = {x for seq in self.said for x in seq[:-1]}
        self.weak = [s for s in views if s not in self.read]
        log(f"minimal-pair signatures (two-link stories only): {len(groups)} groups, {len(big)} with at least "
            f"{a.least} symbols (sizes {' '.join(str(len(groups[g])) for g in big)}); {len(self.read)} symbols grouped, "
            f"{len(self.weak)} not")
        # how the groups sit against the one-link answers (a check only, not used)
        alone = defaultdict(Counter)
        for seq, c in self.said.items():
            if len(seq) == 2 and seq[0] in self.read:
                alone[self.read[seq[0]]].update(c)
        log("  check only - one-link answers by group: " + "; ".join(
            f"{g}: {100 * c.most_common(1)[0][1] / sum(c.values()):.1f}% one answer" for g, c in sorted(alone.items())))
        self.check_pairs(log)
        placed = self.place(self.weak, log)
        self.read.update(placed)
        log(f"placed by minimal pairs: {len(placed)} of {len(self.weak)}")

    def gseq(self, seq, read=None):
        read = read or self.read
        if any(x not in read for x in seq[:-1]):
            return None
        return tuple(read[x] for x in seq[:-1]) + ("Q",)

    def check_pairs(self, log):
        bucket = defaultdict(list)
        for seq, y in self.every.items():
            for i in range(len(seq) - 1):
                bucket[(seq[:i], seq[i + 1:])].append((seq[i], y))
        stat = Counter()
        for (l, r), xs in bucket.items():
            if len(l) + len(r) != 2:      # one other link besides the question
                continue
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    (p, yp), (q, yq) = xs[i], xs[j]
                    if p in self.read and q in self.read:
                        stat[(self.read[p] == self.read[q], yp == yq)] += 1
        for same in (True, False):
            n = stat[(same, True)] + stat[(same, False)]
            log(f"  minimal pairs in two-link stories, the two symbols {'in one group' if same else 'in different groups'}: "
                f"{n} pairs, the answer kept in {100 * stat[(same, True)] / max(1, n):.1f}%")

    def place(self, weak, log):
        """a symbol not grouped: put in place of it every group, in every teaching story it stands in (its other
        links grouped); the group whose stories have the same answer most often (minimal pairs) is its reading"""
        gl = defaultdict(Counter)
        for seq, c in self.said.items():
            g = self.gseq(seq)
            if g:
                gl[g].update(c)
        glmaj = {g: c.most_common(1)[0][0] for g, c in gl.items()}
        out = {}
        for s in weak:
            score = Counter()
            for seq, y in self.every.items():
                for i, x in enumerate(seq[:-1]):
                    if x != s:
                        continue
                    rest = [self.read.get(z) for j, z in enumerate(seq[:-1]) if j != i]
                    if None in rest:
                        continue
                    for G in self.groups:
                        g = tuple(rest[:i]) + (G,) + tuple(rest[i:]) + ("Q",)
                        if g in glmaj:
                            score[G] += 1 if glmaj[g] == y else -1
            if score and score.most_common(1)[0][1] > 0:
                out[s] = score.most_common(1)[0][0]
        return out

    # ------------------------------------------------------------ 2-3. the machine on group symbols, shape found too
    def machine(self, log):
        a = self.a
        gl = defaultdict(Counter)
        for seq, c in self.said.items():
            g = self.gseq(seq)
            if g:
                gl[g].update(c)
        self.gl = gl
        self.glmaj = {g: c.most_common(1)[0][0] for g, c in gl.items()}
        clean = {g: c.most_common(1)[0][0] for g, c in gl.items() if sum(c.values()) >= 2 and len(c) == 1}
        log(f"group-level stories: {len(gl)} different; clean (told at least twice, one answer): {len(clean)}")
        syms = self.groups + ["Q"]
        rng = random.Random(a.seed)
        pool = sorted(clean)
        data = [(g, clean[g]) for g in (rng.sample(pool, a.teach) if 0 < a.teach < len(pool) else pool)]
        if a.shape == "found":
            self.m, asked = cells.learn(syms, lambda w: clean.get(tuple(w)), data, rng, a.asks, a.seconds,
                                        5, quiet=False, pool=pool, workers=a.workers)
        else:
            shape = (0, 1, (("Z", 1, 1), ("Z", 1, 1)), True, False)
            self.m, how = cells.fit(shape, syms, sorted({y for _, y in data}), data, a.seconds * 10)
            asked = 0
        self.asked_n = asked
        if self.m is None:
            raise SystemExit("no machine found")
        log(f"machine found by the solver: {self.m}; taught on {len(data)} group-level stories, asked {asked}")
        log("\n" + self.m.table())
        right = sum(self.m.run(g) == y for g, y in clean.items())
        log(f"  it answers {right} of {len(clean)} clean group-level stories rightly, "
            f"{sum(self.m.run(g) == y for g, y in self.glmaj.items())} of {len(self.glmaj)} group-level stories")
        self.lex = None

    # ------------------------------------------------------------ 4. the solver corrects the reading
    def correct(self, log):
        """cells.fit with lexicon: the raw symbols are the groups (each one raw symbol), Q and the symbols whose
        signature is weak; the solver finds the machine and the reading of all of them at once. Taught greedily from
        the clean stories: stories the machine answers wrongly are added a few at a time, set aside if none fits."""
        a = self.a
        shape = self.m.shape
        weak = [s for s in self.weak]
        alias = {s: f"W{i}" for i, s in enumerate(weak)}
        read = dict(self.read)
        for s in weak:
            read[s] = alias[s]
        raw = self.groups + ["Q"] + [alias[s] for s in weak]
        data = {}
        for seq, c in self.said.items():
            g = self.gseq(seq, read)
            data.setdefault(g, Counter()).update(c)
        data = [(g, c.most_common(1)[0][0]) for g, c in data.items()]
        clean = [(g, y) for g, y in data if sum(self.gl.get(g, {}).values()) >= 2 and len(self.gl.get(g, {})) == 1]
        rng = random.Random(a.seed)
        part = rng.sample(clean, a.teach) if 0 < a.teach < len(clean) else list(clean)
        m = len(set(self.m.syms))
        labels = self.m.labels
        log(f"correcting the reading: {len(raw)} raw symbols ({len(weak)} weak ones), lexicon {m}, {len(data)} stories")
        got, how = cells.fit(shape, raw, labels, part, a.seconds * 6, lexicon=m)
        aside = []
        rounds = 0
        while got is not None and time.time() - T0 < a.budget:
            wrong = [x for x in data if x not in aside and got.run(x[0]) != x[1]]
            log(f"  {len(part)} stories explained, {len(aside)} set aside, {len(wrong)} answered wrongly ({how})")
            if not wrong:
                break
            more = rng.sample(wrong, min(a.batch, len(wrong)))
            new, how = cells.fit(shape, raw, labels, part + more, a.seconds * 3, lexicon=m)
            if new is not None:
                got, part = new, part + more
                continue
            for x in more:
                new, how2 = cells.fit(shape, raw, labels, part + [x], a.seconds * 3, lexicon=m)
                if new is not None:
                    got, part = new, part + [x]
                else:
                    aside.append(x)
            rounds += 1
        if got is None:
            log(f"  no corrected machine ({how}); the first machine kept")
            return
        self.m2, self.read2, self.aside = got, read, aside
        moved = Counter((self.read[s] if s in self.read else "-", got.lex.get(alias[s])) for s in weak)
        log(f"  corrected machine: {got}; reading of groups {dict((g, got.lex[g]) for g in self.groups + ['Q'])}")
        log("\n" + got.table())

    def check(self, log):
        """the reading corrected by exact check against the machine found: every symbol (the weak ones first) is
        read as the symbol of the machine that answers rightly the most teaching stories it stands in (the other
        symbols as they are read); two passes. The machine is not changed."""
        a = self.a
        where = defaultdict(list)
        for seq, c in self.said.items():
            for x in set(seq[:-1]):
                where[x].append((seq, c))
        msyms = self.groups
        order = list(self.weak) + sorted((s for s in where if s not in self.weak), key=lambda s: len(where[s]))
        moved = Counter()
        for rnd in range(a.passes):
            changed = 0
            for s in order:
                best, old = None, self.read.get(s)
                for G in msyms:
                    self.read[s] = G
                    score = 0
                    for seq, c in where[s]:
                        g = self.gseq(seq)
                        if g is not None:
                            score += c[self.m.run(g)]
                    if best is None or score > best[0] or (score == best[0] and G == old):
                        best = (score, G)
                if best[0] == 0 and old is None:
                    self.read.pop(s, None)
                    continue
                self.read[s] = best[1]
                if best[1] != old:
                    changed += 1
                    moved[("weak" if s in self.weak else "grouped")] += 1
            log(f"  exact check, pass {rnd + 1}: {changed} symbols read otherwise")
        log(f"  read otherwise in all: {dict(moved)}; symbols read: {len(self.read)}")

    # ------------------------------------------------------------ running
    def run(self, seq):
        if getattr(self, "m2", None) is not None:
            g = self.gseq(seq, self.read2)
            return None if g is None else self.m2.run(g)
        g = self.gseq(seq)
        return None if g is None else self.m.run(g)

    def answer(self, story):
        seq = seq_of(story)
        return None if seq is None else self.run(seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stepgame", default=DATA)
    p.add_argument("--sure", type=float, default=0.8)
    p.add_argument("--signature", default="alone", choices=["alone", "pairs"],
                   help="alone: the answers of the stories where a symbol is the only link; pairs: minimal pairs in two-link stories only")
    p.add_argument("--evidence", type=int, default=5)
    p.add_argument("--least", type=int, default=3)
    p.add_argument("--shape", default="found", choices=["found", "register"])
    p.add_argument("--teach", type=int, default=0, help="group-level stories to teach on (0: all the clean ones)")
    p.add_argument("--asks", type=int, default=30)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--correct", type=int, default=2, help="0: none, 1: solver with lexicon, 2: exact check")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--budget", type=float, default=900)
    p.add_argument("--save", default="")
    a = p.parse_args(ARGV[1:])
    log(vars(a))
    taught, tried = one.stepgame(a.stepgame)
    log("loaded")
    model = Signature(taught, a)
    rng = random.Random(1)
    acc, none = [], []
    for k in range(1, 11):
        some = rng.sample(tried[f"k={k}"], 1000)
        got = [model.answer(s) for s in some]
        acc.append(100 * sum(g == s[4] for g, s in zip(got, some)) / len(some))
        none.append(100 * sum(g is None for g in got) / len(some))
    log("test, answered rightly %: " + " ".join(f"k={k}: {x:.1f}" for k, x in enumerate(acc, 1)))
    log("test, no answer %:       " + " ".join(f"k={k}: {x:.1f}" for k, x in enumerate(none, 1)))
    out = dict(acc=acc, none=none, explains=model.explains, explains_all=model.explains_all, different=len(model.every),
               asked=model.asked_n, aside=len(getattr(model, "aside", [])), seconds=time.time() - T0,
               machine=str(model.m), table=(getattr(model, "m2", None) or model.m).table())
    print("RESULT", out, flush=True)
    if a.save:
        pickle.dump(out, open(a.save, "wb"))


if __name__ == "__main__":
    main()
