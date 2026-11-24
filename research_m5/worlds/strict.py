"""A stricter check of a machine against the rule of a world: every story up to a length (all of them), and long
stories that keep away from an answer the rule never leaves once given (a broken bracket stays broken, a "no" of
(10)* stays "no"); on random long stories such an answer is almost always the answer, and a machine that always
gives it scores near 100 %.

    python worlds/strict.py WORLDS_FOLDER MACHINES_FOLDER
"""
import importlib.util, itertools, os, pickle, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    spec = importlib.util.spec_from_file_location("family", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.ALPHABET), lambda w: str(mod.answer(list(w)))


def sink(rule, syms, w, rng, tries=6, deep=8):
    """Whether the answer after w looks final: every one of a few random continuations keeps it all the way."""
    y = rule(w)
    for _ in range(tries):
        v = list(w)
        for _ in range(deep):
            v.append(rng.choice(syms))
            if rule(v) != y:
                return False
    return True


def alive(rule, syms, n, rng):
    """A story of n symbols, each chosen among those after which the answer is not final (any, if none is)."""
    w = []
    for _ in range(n):
        ok = [s for s in syms if not sink(rule, syms, w + [s], rng)]
        w.append(rng.choice(ok or syms))
    return w


def check(m, syms, rule, rng, n_long=60):
    short = {2: 14, 3: 9, 4: 8}.get(len(syms), 7)
    total = right = 0
    for n in range(1, short + 1):
        for w in itertools.product(syms, repeat=n):
            total += 1
            right += m.run(list(w)) == rule(w)
    out = {"short": (right, total, short)}
    for L in (100, 500):
        ws = [alive(rule, syms, L, rng) for _ in range(n_long)]
        out[L] = 100 * sum(m.run(w) == rule(w) for w in ws) / len(ws)
        out[f"common{L}"] = 100 * max(sum(rule(w) == y for w in ws) for y in {rule(w) for w in ws}) / len(ws)
        # and the step into a final answer, deep in a long story (a wrong bracket after many open ones)
        into = []
        for w in ws:
            for _ in range(20):
                cut = rng.randint(L // 2, L - 1)
                last = [s for s in syms if sink(rule, syms, w[:cut] + [s], rng)]
                if last:
                    into.append(w[:cut] + [rng.choice(last)])
                    break
        out[f"into{L}"] = 100 * sum(m.run(v) == rule(v) for v in into) / len(into) if into else None
    return out


def main():
    worlds, machines = sys.argv[1], sys.argv[2]
    import cells, parts  # (the classes of the machines pickled)
    for name in sorted(os.listdir(worlds)):
        p = os.path.join(machines, name + ".pkl")
        if not os.path.exists(os.path.join(worlds, name, "family.py")):
            continue
        if not os.path.exists(p):
            print(f"{name:22s} no machine")
            continue
        m = pickle.load(open(p, "rb"))
        syms, rule = load(os.path.join(worlds, name, "family.py"))
        r = check(m, syms, rule, random.Random(5))
        right, total, short = r["short"]
        print(f"{name:22s} every story up to {short:2d}: {right}/{total} ({100 * right / total:6.2f} %)   long stories kept alive, "
              f"length 100 / 500: {r[100]:5.1f} / {r[500]:5.1f} %  (the commonest answer there: {r['common100']:.0f} / {r['common500']:.0f} %)" +
              ("" if r["into100"] is None else f"   stepping at last into a final answer, 100 / 500: {r['into100']:5.1f} / {r['into500']:5.1f} %"), flush=True)


if __name__ == "__main__":
    main()
