#!/usr/bin/env python3
"""
A network that reads the stories of a new task and says at once how its sentences are to be read.

The learner finds what a kind of sentence means by search: minutes to hours for a task, nearly all of it spent on
readings far from the right one. With the readings given, the same task is learnt in a tenth of a second. Here they
are guessed by a network taught on dreams (relive.py): tasks already learnt, lived again in new stories, and their
variations. It sees no words - only which kinds of sentences there are, which things stand in them, and what the
answers were - and for every kind of sentence says the readings to try first. The ground checks them exactly, as it
checks any guess, and mends them; where nothing fits, the search is run as before.

The same network is taught a second thing, to compare: to answer the questions of a story itself, with no ground -
the stories before it with their answers are all it has of the task. That is what a network grown on the ground's
dreams can do by itself; the first is what it can do when the ground executes and checks.

    python guess.py --lived lived.pkl --hold H,I,J --minutes 90 --out guess.pt     # taught without the sets H, I, J; tried on them
    python guess.py --lived lived.pkl --net guess.pt --hold H,I,J                   # only tried
"""
import argparse
import itertools
import math
import os
import pickle
import random
import time

import torch
from torch import nn

import memory
import records
import relive

PAD, STORY, ANS, END, OTHER = 0, 1, 2, 3, 4
PLAIN = {"yes": 5, "no": 6, "maybe": 7, "nothing": 8}
HID = 9                                                         # stands where an answer is not shown: the network is to give it
STATED, ASKED, KINDS, EACH, NUMS = 40, 24, 8, 20, 41            # how many kinds of sentences, questions, kinds of things, things of a kind, numbers
S0 = 10
Q0 = S0 + STATED
T0 = Q0 + ASKED
N0 = T0 + KINDS * EACH
VOCAB = N0 + NUMS


def canon(rec, order):
    """the readings with their stores numbered as they come (the first store of a sort met is 0): one name for the
    readings that differ only in which store is which"""
    names, out = {}, {}
    for f in order:
        new = []
        for name, k, pick, n in rec.get(f, ()):
            sort = records.MOVES[name][1]
            mine = names.setdefault(sort, {})
            new.append((name, mine.setdefault(k, len(mine)), tuple(pick), n))
        out[f] = tuple(new)
    return out


def order(episodes):
    """the kinds of statements and of questions, as they come in the stories"""
    said, asked = [], []
    for ep in episodes:
        for op, quest, _, _, kinds, _ in ep:
            f = (op, quest, kinds)
            if f not in (asked if quest else said):
                (asked if quest else said).append(f)
    return said, asked


def tokens(episodes, rng=None, most=1536, hide=0):
    """stories -> (the tokens; the kinds of statements in the order they are asked about at the end; where the answers
    not shown stand and what they are), or None if the task has more kinds of sentences than the network has names
    for. rng: the names are dealt anew (in teaching); hide: the answers of so many last stories are not shown"""
    said, asked = order(episodes)
    if len(said) > STATED or len(asked) > ASKED:
        return None
    sid, qid = list(range(STATED)), list(range(ASKED))
    kid, eid = list(range(KINDS)), {}
    if rng:
        rng.shuffle(sid), rng.shuffle(qid), rng.shuffle(kid)
    code, kinds, parts = {}, {}, []

    def thing(x):
        if x not in code:
            k = kinds.setdefault(x[0], kid[len(kinds) % KINDS])
            if k not in eid:
                eid[k] = list(range(EACH))
                if rng:
                    rng.shuffle(eid[k])
            code[x] = T0 + k * EACH + eid[k][sum(1 for y in code if kinds[y[0]] == k) % EACH]
        return code[x]

    for n, ep in enumerate(episodes):
        out, asks = [STORY], []
        for op, quest, things, numbers, ks, gold in ep:
            f = (op, quest, ks)
            out.append(Q0 + qid[asked.index(f)] if quest else S0 + sid[said.index(f)])
            ts, ns = iter(things), iter(numbers)
            out += [N0 + min(NUMS - 1, max(0, next(ns))) if k == "#" else thing(next(ts)) for k in ks]
            if quest:
                told = [PLAIN[x] if x in PLAIN else N0 + min(NUMS - 1, int(x)) if x.isdigit() else thing(x) if x[:1].isalpha() and x[1:].isdigit() else OTHER
                        for x in gold.split(",")]
                if n >= len(episodes) - hide:
                    asks.append((len(out), frozenset(told)))
                    out.append(HID)
                else:
                    out += [ANS] + told
        parts.append((out, asks))
    while len(parts) > 1 and sum(len(p) for p, _ in parts) + 1 + len(said) > most:      # too long: the first stories are left out
        parts.pop(0)
    if sum(len(p) for p, _ in parts) + 1 + len(said) > most:
        return None
    out, asks = [], []
    for p, mine in parts:
        asks += [(len(out) + i, told) for i, told in mine]
        out += p
    return out + [END] + [S0 + sid[i] for i in range(len(said))], said, asks


class Net(nn.Module):
    def __init__(self, classes, width=192, layers=6, heads=6, most=1536, loops=1):
        super().__init__()
        self.most, self.loops = most, loops                      # loops: the same layers gone through so many times
        self.tok, self.pos = nn.Embedding(VOCAB, width, padding_idx=PAD), nn.Embedding(most, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, "gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)                    # at the name of a kind of sentence after the stories: how it is read
        self.say = nn.Linear(width, VOCAB)                       # where an answer is not shown: the words of the answer
        nn.init.constant_(self.say.bias, -5.0)                   # (nearly every word is not in an answer)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for _ in range(self.loops):
            h = self.body(h, src_key_padding_mask=x == PAD)
        return self.norm(h)


def batch(items, device):
    """[(tokens, the class of every statement asked about at the end, the answers not shown)] -> the tokens; where
    the readings are named and what they are; where the answers are to be given and what they are"""
    width = max(len(t) for t, _, _ in items)
    x = torch.zeros(len(items), width, dtype=torch.long)
    at, y, where, told = [], [], [], []
    for i, (t, c, asks) in enumerate(items):
        x[i, :len(t)] = torch.tensor(t)
        at += [(i, len(t) - len(c) + j) for j in range(len(c))]
        y += c
        for j, words in asks:
            where.append((i, j))
            told.append(sorted(words))
    at, where = torch.tensor(at), torch.tensor(where or [(0, 0)])
    z = torch.zeros(len(told), VOCAB)
    for n, words in enumerate(told):
        z[n, words] = 1
    return x.to(device), (at[:, 0].to(device), at[:, 1].to(device), torch.tensor(y).to(device)), (where[:, 0].to(device), where[:, 1].to(device), z.to(device))


class Dreams(torch.utils.data.IterableDataset):
    """the stream of dreams as the network takes them: (tokens, the class of every kind of statement, the answers of
    the last stories - they are not shown)"""

    def __init__(self, tasks, classes, a, turn=0):
        self.tasks, self.classes, self.a, self.turn = tasks, classes, a, turn

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(1000003 * self.a.seed + 1009 * self.turn + (info.id if info else 0))
        for eps, rec, _ in relive.dreams(self.tasks, rng, (min(self.a.least, self.a.stories), self.a.stories), self.a.varied):
            got = tokens(eps, rng, self.a.most, rng.choice((1, 1, 2, 3)) if len(eps) > 4 else 0)
            if got:
                rec = canon(rec, got[1])
                yield got[0], [self.classes.get(rec[f], 0) for f in got[1]], got[2]


def keep(path, net, opt, readings, a, spent, step, seen, names):
    """what is learnt so far, written so that a run cut short goes on from here"""
    torch.save({"weights": net.state_dict(), "opt": opt.state_dict(), "readings": readings, "size": (a.width, a.layers, a.heads, a.most, a.loops),
                "spent": spent, "step": step, "seen": seen, "taught on": names}, path + ".part")
    os.replace(path + ".part", path)


def teach(tasks, readings, a, device, log=print):
    """the network taught on dreams of the tasks for so many minutes (a run cut short goes on from what it wrote) -> the network"""
    torch.manual_seed(a.seed)
    classes = {moves: n for n, moves in enumerate(readings)}
    net = Net(len(classes), a.width, a.layers, a.heads, a.most, a.loops).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    before, step, seen = 0.0, 0, 0
    if a.out and os.path.exists(a.out):
        saved = torch.load(a.out, map_location=device)
        net.load_state_dict(saved["weights"])
        opt.load_state_dict(saved["opt"])
        before, step, seen = saved["spent"], saved["step"], saved["seen"]
        log(f"goes on from {before:.1f} min, step {step}")
    stream = iter(torch.utils.data.DataLoader(Dreams(tasks, classes, a, step), batch_size=a.batch, num_workers=a.workers, collate_fn=list))
    t0, last, avg = time.time(), time.time(), None
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    names = [t["name"] for t in tasks]
    while before + (time.time() - t0) / 60 < a.minutes:
        x, (i, j, y), (qi, qj, z) = batch(next(stream), device)
        done = (before + (time.time() - t0) / 60) / a.minutes
        for group in opt.param_groups:
            group["lr"] = a.lr * min(1, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, done))))
        with torch.autocast(device, enabled=amp):
            h = net(x)
            out, told = net.head(h[i, j]).float(), net.say(h[qi, qj]).float()
            loss = nn.functional.cross_entropy(out, y)
            if z.shape[0]:
                loss = loss + nn.functional.binary_cross_entropy_with_logits(told, z, reduction="sum") / z.shape[0]
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step, seen = step + 1, seen + x.shape[0]
        said = (told > 0) | (nn.functional.one_hot(told.argmax(1), VOCAB) > 0) & ~(told > 0).any(1, keepdim=True)     # (the words above the line, or the likeliest one)
        now = [loss.item(), (out.argmax(1) == y).float().mean().item(), (said == (z > 0)).all(1).float().mean().item() if z.shape[0] else 0.0]
        avg = now if avg is None else [0.98 * u + 0.02 * v for u, v in zip(avg, now)]
        if step % a.every == 0:
            log(f"step {step}, {seen} dreams, {before + (time.time() - t0) / 60:.1f} min: loss {avg[0]:.3f}; the reading named first is the right one for "
                f"{100 * avg[1]:.0f}% of the kinds of sentences; answers given right by the network itself {100 * avg[2]:.0f}%")
        if a.out and time.time() - last > 60 * a.save:
            keep(a.out, net, opt, readings, a, before + (time.time() - t0) / 60, step, seen, names)
            last = time.time()
    del stream
    if a.out:
        keep(a.out, net, opt, readings, a, a.minutes, step, seen, names)
    return net


def answers(net, shown, later, device, most=40):
    """the network answering by itself: the stories shown with their answers, then a later story with its answers not
    shown - and so for every later story (at most so many) -> answers right, all"""
    right = total = 0
    net.eval()
    for ep in later[:most]:
        got = tokens(shown + [ep], None, net.most, 1)
        if not got or not got[2]:
            continue
        with torch.no_grad():
            h = net(torch.tensor([got[0]], device=device))[0]
            out = net.say(h[[j for j, _ in got[2]]]).float().cpu()
        for n, (_, words) in enumerate(got[2]):
            mine = set((out[n] > 0).nonzero().flatten().tolist()) or {out[n].argmax().item()}
            right += mine == set(words)
            total += 1
    return right, total


def suits(moves, form):
    """can a kind of sentence of this build be read so: every thing and number the moves take is there"""
    nt, nn = records.size(form)
    for name, k, pick, n in moves:
        if n is not None and n >= nn:
            return False
        for i in pick:
            if not (0 <= i < nt or i in (-1, -2) or (i <= -10 and -10 - i < nn) or (i >= 100 and (i - 100) // 10 < nt and (i - 100) % 10 < nt)):
                return False
    return True


def propose(net, readings, episodes, device, keep=8, wide=5):
    """the stories of a task -> the readings of all its kinds of sentences to try, the likeliest first (each a whole
    reading of the task: {kind of sentence: moves}); the kinds of sentences; and for every kind of sentence apart the
    readings the network puts first"""
    got = tokens(episodes, None, net.most)
    if not got:
        return [], [], []
    toks, said, _ = got
    net.eval()
    with torch.no_grad():
        out = net.head(net(torch.tensor([toks], device=device))[0, len(toks) - len(said):]).float().cpu()
    for n, f in enumerate(said):                                 # (only the readings the build of the sentence allows)
        out[n, [c for c, moves in enumerate(readings) if not suits(moves, f)]] = -1e9
    out = torch.log_softmax(out, 1)
    best = [[(out[n, c].item(), c) for c in out[n].topk(min(wide, out.shape[1])).indices.tolist()] for n in range(len(said))]
    whole = [(0.0, ())]                                          # the likeliest whole readings: a beam over the kinds of sentences
    for options in best:
        whole = sorted(((p + q, picks + (c,)) for p, picks in whole for q, c in options[:3]), reverse=True)[:4 * keep]
    return [{f: readings[c] for f, c in zip(said, picks)} for _, picks in whole[:keep]], said, [[readings[c] for _, c in options] for options in best]


def mend(m, rec, said, options, sweeps=3, every=False):
    """a reading proposed, mended by the ground: for every kind of sentence in turn the other readings the network
    puts first are tried in its place, and the one that explains more is kept - until every question is explained or
    nothing helps. every: after that, for a kind of sentence at a time, every reading its build allows. A search
    still, but of dozens (every: hundreds) of exact checks -> the reading, the questions it explains"""
    total = sum(len(g) for f, g in m.forms.items() if f[1])
    rec = {f: rec.get(f, ()) for f in m.rec}

    def worth(r):
        try:
            return m.fit(r)[0]
        except (KeyError, IndexError, ValueError, TypeError):
            return (-1.0, 0)
    best = worth(rec)
    for wide in (False, True) if every else (False,):
        for _ in range(sweeps):
            moved = False
            for f, mine in zip(said, options):
                if wide:
                    top = {s: max([k for moves in rec.values() for name, k, _, _ in moves if records.MOVES[name][1] == s] + [-1]) + 1 for s in records.K}
                    mine = [p for p in m.rows(*records.size(f), top, "FQEL") if p not in mine]
                for moves in mine:
                    if best[1] == total:
                        return rec, best[1]
                    if f in rec and moves != rec[f]:
                        new = {**rec, f: moves}
                        got = worth(new)
                        if got > best:
                            rec, best, moved = new, got, True
            if not moved:
                break
    return rec, best[1]


LOADED = {}


def proposed(path, episodes, keep=8):
    """the readings a network kept in a file proposes for these stories: the best of them mended by the ground, then
    the others as the network gave them, the likeliest first"""
    if path not in LOADED:
        saved = torch.load(path, map_location="cpu")
        net = Net(len(saved["readings"]), *saved["size"])
        net.load_state_dict(saved["weights"])
        LOADED[path] = (net, saved["readings"])
    recs, said, options = propose(*LOADED[path], episodes, "cpu", keep)
    if not recs:
        return []
    right, total, rec, m = explained(episodes, recs)
    return [mend(m, rec or recs[0], said, options, every=True)[0]] + recs


def explained(episodes, recs):
    """the readings tried on the stories by the ground itself -> (questions answered right, all questions, the best reading)"""
    m = records.Learner(episodes)
    best = (0, None)
    for rec in recs:
        rec = {f: rec.get(f, ()) for f in m.rec}
        try:
            right = m.fit(rec)[0][1]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if right > best[0]:
            best = (right, rec)
    return best[0], sum(len(g) for f, g in m.forms.items() if f[1]), best[1], m


def held(m, rec, later):
    """what a reading taken on the first stories gives on the later ones -> right, all"""
    m.rec = rec
    m.ask = m.fit(rec)[1]
    m.usual = {f: records.Counter(g).most_common(1)[0][0] for f, g in m.forms.items() if f[1]}
    return m.run(later)


def trial(net, readings, tasks, known, device, stories=20, log=print):
    """the network on tasks it was not taught on. For every task: its first stories, the readings proposed, the exact
    check by the ground; beside it, what is remembered word for word (known: the readings of the kinds of sentences
    of the tasks taught on)"""
    rows = []
    for t in tasks:
        eps, later = t["episodes"][:stories], t["episodes"][stories:]
        t0 = time.time()
        recs, said, options = propose(net, readings, eps, device)
        total = sum(r[1] for ep in eps for r in ep)
        first = explained(eps, recs[:1])
        some = explained(eps, recs)
        mem = explained(eps, [memory.recall(known, said)])
        start = max(some, mem, key=lambda x: x[0])
        rec, right = mend(start[3], start[2] or {}, said, options) if recs else (start[2], start[0])
        t1 = time.time()
        wide = mend(start[3], rec, said, options, every=True) if recs and right < total else (rec, right)
        t2 = time.time() - t1
        on = held(start[3], wide[0], later) if wide[0] else (0, sum(r[1] for ep in later for r in ep))
        true = canon(t["rec"], said)
        same = sum(canon(recs[0], said)[f] == true[f] for f in said) if recs and t["full"] else None
        among = sum(any(canon({f: o}, [f])[f] == canon({f: t["rec"][f]}, [f])[f] for o in mine) for f, mine in zip(said, options)) if recs and t["full"] else None
        own = answers(net, eps, later, device)
        rows.append((t["name"], t["full"], len(said), same, among, first[0], some[0], mem[0], right, total, on, t1 - t0, wide[1], t2, own))
        log(f"{t['name']:5} {'learnt' if t['full'] else 'not learnt':10} kinds of sentences {len(said):2}" + (f" (named right first {same}, among the first five {among})" if same is not None else "")
            + f"; its questions explained: the first guess {first[0]}/{total}, the best of {len(recs)} {some[0]}/{total}, by memory {mem[0]}/{total}, mended {right}/{total} ({t1 - t0:.1f} s), "
              f"mended wide {wide[1]}/{total} ({t2:.0f} s); on the later stories {on[0]}/{on[1]}; the network answering by itself {own[0]}/{own[1]}")
    for label, pick in (("tasks the learner had learnt in full", [r for r in rows if r[1]]), ("tasks it had not", [r for r in rows if not r[1]])):
        if pick:
            n, every = len(pick), sum(r[9] for r in pick)
            kinds = sum(r[2] for r in pick)
            log(f"{label}: {n}. Every question of the first stories explained - by the first guess in {sum(r[5] == r[9] for r in pick)} tasks, by one of eight in {sum(r[6] == r[9] for r in pick)}, "
                f"by memory word for word in {sum(r[7] == r[9] for r in pick)}, after mending in {sum(r[8] == r[9] for r in pick)}, after mending wide in {sum(r[12] == r[9] for r in pick)}; "
                f"questions explained {100 * sum(r[5] for r in pick) / every:.1f}% (first guess), {100 * sum(r[7] for r in pick) / every:.1f}% (memory), {100 * sum(r[8] for r in pick) / every:.1f}% (mended), "
                f"{100 * sum(r[12] for r in pick) / every:.1f}% (mended wide); on the later stories {100 * sum(r[10][0] for r in pick) / max(1, sum(r[10][1] for r in pick)):.1f}%; "
                f"the network answering by itself {100 * sum(r[14][0] for r in pick) / max(1, sum(r[14][1] for r in pick)):.1f}%; "
                f"{sum(r[11] for r in pick) / n:.1f} s a task, mending wide {sum(r[13] for r in pick) / n:.0f} s more"
                + (f"; kinds of sentences named right first {100 * sum(r[3] for r in pick) / kinds:.0f}%, among the first five {100 * sum(r[4] for r in pick) / kinds:.0f}%" if pick[0][1] else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lived", required=True, help="the tasks as they were understood (made by relive.py)")
    ap.add_argument("--hold", default="", help="sets left out of teaching and tried on: H,I")
    ap.add_argument("--net", default="", help="a network already taught")
    ap.add_argument("--out", default="")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--stories", type=int, default=20)
    ap.add_argument("--least", type=int, default=8, help="a dream has from so many stories to --stories")
    ap.add_argument("--varied", type=float, default=0.5, help="how many of the dreams are tasks that never were")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--loops", type=int, default=1, help="the layers are gone through so many times (the same weights): a loop instead of a taller pile")
    ap.add_argument("--most", type=int, default=1536, help="the longest row of tokens")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=2, help="how many processes make up the dreams while the network learns")
    ap.add_argument("--few", type=int, default=0, help="try on only so many of the tasks left out (a short check that everything runs)")
    ap.add_argument("--save", type=float, default=10, help="what is learnt is written every so many minutes; a run started again goes on from it")
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--look", type=int, default=10000, help="how many dreams are looked at first, for the readings the network may name")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = pickle.load(open(a.lived, "rb"))
    hold = tuple(x for x in a.hold.split(",") if x)
    apart = [t for t in tasks if t["name"].rstrip("0123456789") in hold]
    mine = [t for t in tasks if t["full"] and t not in apart]
    known = {}
    for t in mine:
        for f, moves in t["rec"].items():
            if moves and all(m != moves for m, _ in known.setdefault(memory.key(f[0], f[2]), [])):
                known[memory.key(f[0], f[2])].append((moves, t["name"]))
    print(f"{len(mine)} tasks to dream of, {len(apart)} left out to try on; {device}", flush=True)
    if a.net:
        saved = torch.load(a.net, map_location=device)
        readings = saved["readings"]
        net = Net(len(readings), *saved["size"]).to(device)
        net.load_state_dict(saved["weights"])
    else:
        if a.out and os.path.exists(a.out):                      # a run cut short: the same readings as it began with
            readings = torch.load(a.out, map_location="cpu")["readings"]
        else:
            rng = random.Random(a.seed + 1)                      # the readings the network may name: those met in a first look at the dreams
            classes = {(): 0}
            for eps, rec, _ in itertools.islice(relive.dreams(mine, rng, 4, a.varied), a.look):
                for moves in canon(rec, order(eps)[0]).values():
                    classes.setdefault(moves, len(classes))
            readings = sorted(classes, key=classes.get)
        print(f"{len(readings)} readings a kind of sentence may be given", flush=True)
        net = teach(mine, readings, a, device, lambda s: print(s, flush=True))
    print(f"{sum(p.numel() for p in net.parameters())} weights", flush=True)
    if apart:
        trial(net, readings, apart[:a.few or None], known, device, a.stories, lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
