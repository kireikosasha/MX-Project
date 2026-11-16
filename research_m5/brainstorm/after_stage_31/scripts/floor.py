"""How much of 'explained from the words alone' is a floor any reading gets?
For every task of a set, on its 40 train stories, with the same code path as soft.py:
  majority  - the most frequent answer of every kind of question
  empty     - no statement plays any part; of every kind of question the best read, default and turn fitted (grade)
  untrained - a Reader with random weights (mean of 5 seeds)
  masked    - the taught Reader, every word but the place marks made unknown (shape prior only)
  shuffled  - the taught Reader, the texts permuted among kinds of sentences of the same shape
  taught    - the taught Reader (the published number)
  strict    - the taught Reader, no default and no turn fitted to the answers (truly zero answers used... but for
              the guess about words chosen)
"""
import glob, os, sys, random
from collections import Counter
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import torch
import recall, rows, soft

torch.set_num_threads(1)
LOGIC = "/tmp/claude-0/-home-user-MX-Project/ace7a071-a9eb-520e-b8a4-5dce652d6be1/scratchpad/logic"


def load(path):
    saved = torch.load(path)
    r = soft.Reader(saved["vocab"], whole=saved.get("whole", False))
    r.load_state_dict(saved["weights"])
    return r


def hard_score(layer, task, strict=False):
    readings = layer.hard(task)
    right = sum(r.right for r in readings.values())
    total = sum(r.total for r in readings.values())
    if strict:
        right = 0
        for q, r in readings.items():
            pairs = recall.run(task, q, r.sort, r.ask, r.roles, task.own(q))
            for x, g in pairs:
                told = ("yes" if x else "no") if x.__class__ is bool else x
                right += told == g
    return right, total, readings


def empty_score(task, layer):
    right = 0
    for q in layer.nt:
        best = 0
        for sort, name, at in soft.ASKS[layer.shape[q]]:
            if not all(x < layer.nt[q] for x in at):
                continue
            got = recall.grade(recall.run(task, q, sort, (name, at), {}, task.own(q)))
            best = max(best, got[0])
        right += best
    return right


class Masked:
    def __init__(self, reader, mode, seed=0):
        self.reader, self.mode, self.rng = reader, mode, random.Random(seed)

    def choices(self, task, layer):
        r = self.reader
        said = sorted(task.forms.items(), key=lambda x: x[1])
        asked = sorted(task.asked.items(), key=lambda x: x[1])
        texts = [soft.words(f[0], False) for f, _ in said] + [soft.words(f[0], True) for f, _ in asked]
        if self.mode == "masked":
            texts = [[w if w in ("?", ".") or w.startswith("<") else "\x00unk" for w in t] for t in texts]
        elif self.mode == "shuffled":
            groups = {}
            for i, t in enumerate(texts):
                groups.setdefault((t[0], sum(w.startswith("<") for w in t)), []).append(i)
            new = list(texts)
            for idx in groups.values():
                perm = idx[:]
                self.rng.shuffle(perm)
                for a, b in zip(idx, perm):
                    new[a] = texts[b]
            texts = new
        h = r.read(texts)
        hs, hq = h[:len(said)], h[len(said):]
        if r.whole:
            around = torch.cat([hs.mean(0), hq.mean(0)])
            hs, hq = torch.cat([hs, around.expand(len(hs), -1)], 1), torch.cat([hq, around.expand(len(hq), -1)], 1)
        pad = lambda x: torch.cat([x, torch.zeros((1, x.shape[1]))])
        turn = r.heads["turn"](hq)
        return (pad(r.heads["value"](hs)), pad(r.heads["link"](hs)), pad(r.heads["also"](hs)),
                {q: r.heads[layer.shape[q]](hq[i]) for i, (_, q) in enumerate(asked)},
                {q: torch.stack([4.0 + turn[i, 0], turn[i, 1]]) for i, (_, q) in enumerate(asked)})


def main():
    name, wpath = sys.argv[1], sys.argv[2]
    folder = os.path.join(LOGIC, "new", name)
    reader_of_words = load(wpath)
    tot = Counter()
    print(f"set {name}, reader {os.path.basename(wpath)}")
    print("task | total | majority | empty | untrained | masked | shuffled | taught | strict | unknown words (share of the kinds of sentences with one)")
    for k in range(1, 13):
        raw = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], 40)
        later = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*test.txt"))[0])
        tried = []
        for reader in rows.guesses(raw[:20]):
            task = recall.Task(reader.rows(raw), False)
            if max(sum(c != "#" for c in f[2]) for f in list(task.forms) + list(task.asked)) > soft.SLOTS:
                continue
            task = soft.widened(reader, task, later)
            layer = soft.Layer(soft.Stories(task))
            with torch.no_grad():
                layer.given = reader_of_words.choices(task, layer)
                right, total, readings = hard_score(layer, task)
            tried.append((right - 0.5 * len(readings) - 3 * bool(reader.guessed), id(reader), reader, task, layer, right, total))
            if right == total:
                break
        if not tried:
            print(f"{k:4d} | more than three things")
            continue
        _, _, reader, task, layer, right, total = max(tried)
        maj = sum(Counter(task.golds(q)).most_common(1)[0][1] for q in task.asked.values() if task.golds(q))
        emp = empty_score(task, layer)
        with torch.no_grad():
            strict = hard_score(layer, task, strict=True)[0]
            un = []
            for seed in range(5):
                torch.manual_seed(100 + seed)
                fresh = soft.Reader(list(reader_of_words.vocab), whole=reader_of_words.whole)
                layer.given = fresh.choices(task, layer)
                un.append(hard_score(layer, task)[0])
            layer.given = Masked(reader_of_words, "masked").choices(task, layer)
            masked = hard_score(layer, task)[0]
            sh = []
            for seed in range(5):
                layer.given = Masked(reader_of_words, "shuffled", seed).choices(task, layer)
                sh.append(hard_score(layer, task)[0])
        texts = [soft.words(f[0], False) for f in list(task.forms) + list(task.asked)]
        unk = sum(any(w not in reader_of_words.vocab for w in t) for t in texts) / len(texts)
        row = dict(total=total, majority=maj, empty=emp, untrained=sum(un) / 5, masked=masked, shuffled=sum(sh) / 5, taught=right, strict=strict)
        tot.update(row)
        print(f"{k:4d} | {total:4d} | " + " | ".join(f"{100 * row[c] / total:5.1f}" for c in list(row)[1:]) + f" | {unk:.2f}" + (" - " + "; ".join(reader.guessed) if reader.guessed else ""), flush=True)
    print(" sum | {:4d} | ".format(tot["total"]) + " | ".join(f"{100 * tot[c] / tot['total']:5.1f}" for c in ("majority", "empty", "untrained", "masked", "shuffled", "taught", "strict")))


main()
