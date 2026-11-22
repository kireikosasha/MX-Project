"""One shot of the frozen learner (cells.py) on sealed families written by others.

A family is a folder with family.py that defines ALPHABET (a list of token strings) and answer(seq) -> str for any
sequence of tokens. The learner sees what it always sees: 300 sequences of 1-4 tokens with their answers, and may ask
the answers of up to 40 more of its own choosing; it is tested on 300 sequences of length 20, 100 and 500. Nothing in
cells.py was changed for these families; the commit of the code and the hashes of the families are written down before
the run.

    python sealed_run.py FOLDER_OF_FAMILIES [--minutes 30]
"""
import argparse, hashlib, importlib.util, multiprocessing as mp, os, random, sys, time
import cells


def load(path):
    spec = importlib.util.spec_from_file_location("family", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.ALPHABET), mod.answer


def one(path, out, a):
    syms, answer = load(path)
    oracle = lambda w: str(answer(list(w)))
    rng = random.Random(a.seed)
    data = []
    for _ in range(a.teach):
        w = tuple(rng.choice(syms) for _ in range(rng.randint(1, a.longest)))
        data.append((w, oracle(w)))
    t0 = time.time()
    m, asked = cells.learn(syms, oracle, data, rng, a.asks, a.seconds, a.longest, quiet=True, workers=a.workers,
                           shorten=a.shorten, reserve=a.reserve, explore=a.explore, frugal=a.frugal, slack=a.slack, guards="tests" if a.tests else a.guards, build=a.build, plan=a.plan, committee=a.committee, flags=a.flags, cascades=a.cascades, rules=a.rules, keep=a.keep, watch=a.watch, dep=a.dep, warm=a.warm, patience=a.patience, tier=a.tier, tour_=a.tour, anchor=a.anchor, far=a.far, deep=a.deep, mealy=a.mealy,
                           deadline=time.time() + a.minutes * 60 - 180 if a.deadline else None)
    if a.save:  # (the machine itself, for a check that needs more than its answers: external/mealy_dot.py --same)
        import pickle
        os.makedirs(a.save, exist_ok=True)
        pickle.dump(m, open(os.path.join(a.save, os.path.basename(os.path.dirname(path)) + ".pkl"), "wb"))
    res = {"found": None if m is None else str(m), "asked": asked, "seconds": round(time.time() - t0),
           "sure": None if m is None else m.sure}
    for L in (20, 100, 500):
        tr = random.Random(1000 + L)
        seqs = [[tr.choice(syms) for _ in range(L)] for _ in range(300)]
        gold = [oracle(s) for s in seqs]
        res[L] = None if m is None else 100 * sum(m.run(s) == g for s, g in zip(seqs, gold)) / len(seqs)
        res[f"most{L}"] = 100 * max(gold.count(g) for g in set(gold)) / len(gold)   # the share of the commonest answer
    out.put(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--minutes", type=float, default=30, help="the time one family may take before it counts as not taken")
    ap.add_argument("--teach", type=int, default=300)
    ap.add_argument("--longest", type=int, default=4)
    ap.add_argument("--asks", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slack", type=float, default=8, help="a second machine is looked for among the shapes at most so many bits longer")
    ap.add_argument("--warm", action="store_true", help="the solver of every shape is kept from one search to the next (what it learned stays; a second machine is looked for under an assumed literal, on short stories first); one worker")
    ap.add_argument("--patience", type=float, default=0, help="when the search for a second machine has taken so many seconds of the solver's limit without one, one story of the check is asked meanwhile")
    ap.add_argument("--part", type=int, default=0, help="a search begins with so many of the stories (40 if not given) and adds those its machine answers wrongly; a large number: all of them at once")
    ap.add_argument("--tier", type=float, default=0, help="so many bits are added to the price of a shape for a second cell that moves apart from the first (and twice as many for moves priced by the rule, which --rules then adds beside the tables)")
    ap.add_argument("--tour", type=int, default=0, help="with --watch: a story asked goes on, up to so many symbols, through the moves of the kept machine that no story heard has made")
    ap.add_argument("--anchor", type=float, default=0, help="the machines whose states are the answers, each with a few hidden values, are tried right after the shapes at most so many bits long")
    ap.add_argument("--far", type=int, default=0, help="with --watch: eight of the twelve plain stories of the check are of up to so many symbols (a long story heard is checked, not put into the formulas)")
    ap.add_argument("--deep", type=int, default=0, help="with --warm and --build: the last search for a second machine writes stories of up to so many symbols")
    ap.add_argument("--zlong", type=int, default=0, help="with --warm: a shape with a whole number is first told only the stories of at most so many symbols; a longer one when its machine answers it wrongly")
    ap.add_argument("--save", default="", help="a folder to write the machines found into (NN-name.pkl)")
    ap.add_argument("--mealy", action="store_true", help="also machines whose answer is of the last move: read from the state before the last symbol and that symbol")
    ap.add_argument("--engine", default="", help="the solver (of pysat's; glucose4 if not given)")
    ap.add_argument("--dep", action="store_true", help="every body at ceilings of its price, each move priced by what it depends on (the state, the read of the first cell)")
    ap.add_argument("--watch", action="store_true", help="the answer is seen after every symbol (of the stories taught on and of every story asked; a story asked is still one question)")
    ap.add_argument("--keep", action="store_true", help="no answer paid for is thrown away (the check stories, the halving's beginnings)")
    ap.add_argument("--rules", action="store_true", help="moves that read the numbers priced by the rule: a move by default and a few exceptions")
    ap.add_argument("--deadline", action="store_true", help="three minutes before the limit the learner stops and tells its last machine, not sure")
    ap.add_argument("--cascades", action="store_true", help="a level or circle driving a second one by its value, the answer read from the second")
    ap.add_argument("--flags", action="store_true", help="a control made of switches (each symbol keeps, clears, sets or flips each)")
    ap.add_argument("--committee", type=int, default=0, help="with --build: up to so many second machines, and the story that splits them most is asked")
    ap.add_argument("--plan", action="store_true", help="the questions kept for the check include those about the unseen reads")
    ap.add_argument("--build", type=int, default=0, help="the second machine is looked for with a story of at most so many symbols that the solver writes itself, not in a pool")
    ap.add_argument("--tests", action="store_true", help="with --guards: also moves by a test of one cell (reads at least a value)")
    ap.add_argument("--guards", action="store_true", help="moves that read the first number (a counter tested for zero)")
    ap.add_argument("--frugal", action="store_true", help="one question per second machine; the check up to the first wrong answer")
    ap.add_argument("--explore", action="store_true", help="the reads no story taught on ends at are asked about")
    ap.add_argument("--reserve", type=int, default=0, help="so many of the questions are kept for the last check")
    ap.add_argument("--shorten", action="store_true", help="a long story answered wrongly is cut to where the machine first goes wrong")
    ap.add_argument("--after", default="", help="a file of results so far: the families in it are not run again")
    a = ap.parse_args()
    if a.engine:
        cells.ENGINE = a.engine
    if a.part:
        cells.PART = a.part
    if a.zlong:
        cells.ZLONG = a.zlong
    names = sorted(d for d in os.listdir(a.folder) if os.path.exists(os.path.join(a.folder, d, "family.py")))
    print(f"{len(names)} sealed families in {a.folder}", flush=True)
    done = set()
    if a.after and os.path.exists(a.after):  # taken up after a stop: the families already written there are skipped
        done = {line.split(" ")[0] for line in open(a.after) if line[:2].isdigit()}
    for d in names:
        if d in done:
            continue
        path = os.path.join(a.folder, d, "family.py")
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        out = mp.get_context("fork").Queue()
        p = mp.get_context("fork").Process(target=one, args=(path, out, a))
        p.start()
        p.join(a.minutes * 60)
        if p.is_alive():
            p.kill()
            print(f"{d} ({digest}): not taken within {a.minutes:.0f} minutes", flush=True)
            continue
        if out.empty():
            print(f"{d} ({digest}): the run failed", flush=True)
            continue
        r = out.get()
        if r["found"] is None:
            print(f"{d} ({digest}): nothing found; asked {r['asked']} ({r['seconds']} s)", flush=True)
        else:
            print(f"{d} ({digest}): {'NOT SURE: ' if r['sure'] is False else ''}{r['found']}; asked {r['asked']}; length 20 / 100 / 500: "
                  f"{r[20]:.1f} / {r[100]:.1f} / {r[500]:.1f} % (the commonest answer: {r['most20']:.0f} / {r['most100']:.0f} / "
                  f"{r['most500']:.0f} %) ({r['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
