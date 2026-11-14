import random
import sys

# Task: ordering of jobs in time. Statements say which job was done
# before or after another; questions ask for the first or last job, a
# before/after relation, or who did the last job. Answers are derived by
# transitive closure over the stated order, and every question is only
# asked when the answer needs at least one chain of two or more facts.

JOBS = ["floor", "window", "car", "fence", "soup", "bread"]
NAMES = ["Ann", "Ben", "Cara", "Dan", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jo"]


def closure(direct):
    rel = set(direct)
    changed = True
    while changed:
        changed = False
        for a, b in sorted(rel):
            for c, d in sorted(rel):
                if b == c and a != d and (a, d) not in rel:
                    rel.add((a, d))
                    changed = True
    return rel


class Story:
    def __init__(self, order, doers):
        self.order = order          # true time order of the jobs (world)
        self.doers = doers          # true doer of each job (world)
        self.pos = {j: i for i, j in enumerate(order)}
        self.direct = set()         # (a, b): a was stated directly before b
        self.mentioned = []         # jobs named so far, in order of mention
        self.doer = {}              # doer of a job, as stated so far
        self.used = set()
        self.asked = set()
        self.lines = []
        self.stmts = 0
        self.qs = 0

    def note(self, job):
        if job not in self.mentioned:
            self.mentioned.append(job)

    def add_statement(self, rng):
        for _ in range(60):
            i, j = sorted(rng.sample(range(len(self.order)), 2))
            a, b = self.order[i], self.order[j]
            if rng.random() < 0.5:
                subj, job = self.doers[a], a
                text = "%s did the %s before the %s." % (subj, a, b)
            else:
                subj, job = self.doers[b], b
                text = "%s did the %s after the %s." % (subj, b, a)
            if text not in self.used:
                break
        else:
            return False
        self.used.add(text)
        self.direct.add((a, b))
        self.note(a)
        self.note(b)
        self.doer[job] = subj
        self.lines.append(text)
        self.stmts += 1
        return True

    def candidates(self):
        m = self.mentioned
        if len(m) < 2:
            return []
        rel = closure(self.direct)

        def indirect(x, y):
            return (x, y) not in self.direct and (y, x) not in self.direct

        def has_indirect(x):
            return any(y != x and indirect(x, y) for y in m)

        out = []
        firsts = [j for j in m if all(k == j or (j, k) in rel for k in m)]
        lasts = [j for j in m if all(k == j or (k, j) in rel for k in m)]
        if firsts and has_indirect(firsts[0]):
            assert firsts[0] == min(m, key=self.pos.get)
            out.append(("Of the jobs named so far, which was done first?", firsts[0]))
        if lasts and has_indirect(lasts[0]):
            assert lasts[0] == max(m, key=self.pos.get)
            out.append(("Of the jobs named so far, which was done last?", lasts[0]))
            if lasts[0] in self.doer:
                assert self.doer[lasts[0]] == self.doers[lasts[0]]
                out.append(("Who did the last job named so far?", self.doer[lasts[0]]))
        for x, y in sorted(rel):
            if indirect(x, y):
                assert self.pos[x] < self.pos[y]
                out.append(("Was the %s done before the %s?" % (x, y), "yes"))
                out.append(("Was the %s done before the %s?" % (y, x), "no"))
        return out

    def ask(self, rng):
        cands = [c for c in self.candidates() if c[0] not in self.asked]
        if not cands:
            return False
        q, ans = rng.choice(cands)
        self.asked.add(q)
        self.lines.append("%s\t%s" % (q, ans.lower()))
        self.qs += 1
        return True


def make_story(rng):
    while True:
        m = rng.randint(3, 5)
        jobs = rng.sample(JOBS, m)
        order = list(jobs)
        rng.shuffle(order)
        doers = {j: rng.choice(NAMES) for j in jobs}
        n_stmt = rng.randint(3, 12)
        n_q = rng.randint(1, 5)
        st = Story(order, doers)
        ok = True
        while st.stmts < n_stmt:
            if st.qs < n_q and st.stmts > 0 and rng.random() < 0.4 and st.ask(rng):
                continue
            if not st.add_statement(rng):
                ok = False
                break
        if not ok:
            continue
        while st.qs < n_q and st.ask(rng):
            pass
        if st.qs >= 1:
            return st.lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            out.append("%d %s" % (i, line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
