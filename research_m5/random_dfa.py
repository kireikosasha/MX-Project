"""Random small automata as families, in the way of the Abbadingo One competition (Lang, Pearlmutter, Price 1998): a
transition table drawn at random, states accepting at random, and only the automata that are what they look like kept.
They are written after the code to be checked is frozen and are not looked at before the run: nobody chose them.

Automaton i (0, 1, ...) is drawn from random.Random(seed + 1000 * i + t) for t = 0, 1, ... until it is kept: n states
(3 + i % 6, that is 3 to 8), k letters (2 if i is even, 3 if odd: a, b, c), every transition to a state drawn evenly,
every state accepting with chance one half, state 0 the start. Kept when every state is reached from the start and no two
states answer alike (the automaton is minimal: it has exactly n states). The answer after every symbol is "yes" when the
state reached accepts, "no" when not.

    python random_dfa.py FOLDER [--count 20] [--seed 8800]
"""
import argparse, os, random


def draw(i, seed):
    n, k = 3 + i % 6, 2 if i % 2 == 0 else 3
    for t in range(10000):
        rng = random.Random(seed + 1000 * i + t)
        delta = [[rng.randrange(n) for _ in range(k)] for _ in range(n)]
        acc = [rng.random() < 0.5 for _ in range(n)]
        seen, todo = {0}, [0]
        while todo:
            q = todo.pop()
            for r in delta[q]:
                if r not in seen:
                    seen.add(r)
                    todo.append(r)
        if len(seen) < n:
            continue
        part = [int(a) for a in acc]  # (Moore: refine by the classes of the successors until nothing splits)
        while True:
            sig = [(part[q],) + tuple(part[r] for r in delta[q]) for q in range(n)]
            names = {s: j for j, s in enumerate(sorted(set(sig)))}
            new = [names[s] for s in sig]
            if len(set(new)) == len(set(part)):
                break
            part = new
        if len(set(part)) == n:
            return n, k, delta, acc
    raise RuntimeError("no automaton kept")


def write(folder, count, seed):
    for i in range(count):
        n, k, delta, acc = draw(i, seed)
        d = os.path.join(folder, f"{i + 1:02d}-random-{n}-states-{k}-letters")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "family.py"), "w").write(
            f'"""A random automaton of {n} states over {k} letters (random_dfa.py, seed {seed}, number {i})."""\n'
            f"ALPHABET = {list('abc'[:k])!r}\nDELTA = {delta!r}\nACCEPT = {acc!r}\n"
            "def answer(seq):\n    q = 0\n    for t in seq:\n        q = DELTA[q][ALPHABET.index(t)]\n"
            "    return 'yes' if ACCEPT[q] else 'no'\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=8800)
    a = ap.parse_args()
    write(a.folder, a.count, a.seed)
