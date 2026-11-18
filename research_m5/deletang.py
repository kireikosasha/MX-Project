"""The tasks of Deletang et al. (ICLR 2023, "Neural Networks and the Chomsky Hierarchy") that have one answer, taken
over from their code (github.com/google-deepmind/neural_networks_chomsky_hierarchy) as it is, and given to the frozen
learner of cells.py. Their networks are taught on lengths 1-40 and tested on 41-500; the score is the accuracy over the
test lengths, the best of 10 seeds.

Here: taught on 300 stories of lengths 1-10; the learner may ask the answers of up to 40 more, chosen by itself from
2000 stories of lengths 1-20 drawn by the same generator (an arbitrary string of symbols is not a question in these
tasks); tested on 400 stories of lengths drawn evenly from 41-500."""
import argparse, random, sys, time
import cells

M = 5  # the modulus of the paper


# ------------------------------------------------ regular
def even_pairs(rng, n):
    s = [rng.randrange(2) for _ in range(n)]
    return [str(x) for x in s], str(sum(a != b for a, b in zip(s, s[1:])) % 2)


def parity_check(rng, n):
    s = [rng.randrange(2) for _ in range(n)]
    return [str(x) for x in s], str(sum(s) % 2)


def cycle_navigation(rng, n):
    s = [rng.randrange(3) for _ in range(n)]
    return [str(x) for x in s], str(sum(x - 1 for x in s) % M)


def evaluate_simple(tokens):
    """modular_arithmetic.py: blanks none; subtraction as adding the inverse; products first; the sum modulo 5."""
    terms, sign, prod = [], 1, None
    for i, t in enumerate(tokens):
        if i % 2 == 0:
            prod = int(t) if prod is None else prod * int(t)
        elif t == "*":
            continue
        else:
            terms.append(sign * prod)
            sign, prod = (1 if t == "+" else -1), None
    terms.append(sign * prod)
    return sum(terms) % M


def modular_arithmetic_simple(rng, n):
    if n % 2 != 1:
        n -= 1
    n = max(n, 1)
    toks = [str(rng.randrange(M)) if i % 2 == 0 else rng.choice("+*-") for i in range(n)]
    return toks, str(evaluate_simple(toks))


# ------------------------------------------------ deterministic context-free
def expression(rng, length, mult):
    """modular_arithmetic_brackets.generate_one_expression_and_result."""
    term = lambda: rng.randrange(M)
    if length == 1:
        t = term(); return str(t), t
    if length == 2:
        t = term(); return f"-{t}", -t % M
    if length == 3:
        t = term(); return f"({t})", t % M
    if length == 4:
        t = term(); return f"(-{t})", -t % M
    left = rng.randrange(1, length - 3)
    right = length - (left + 3)
    ls, lv = expression(rng, left, mult)
    rs, rv = expression(rng, right, mult)
    op = rng.randrange(3 if mult else 2)
    if op == 0:
        return f"({ls}+{rs})", (lv + rv) % M
    if op == 1:
        return f"({ls}-{rs})", (lv - rv) % M
    return f"({ls}*{rs})", (lv * rv) % M


def modular_arithmetic_brackets(rng, n):
    s, v = expression(rng, max(n, 1), mult=True)  # the paper runs it with multiplication
    return list(s), str(v)


def solve_equation(rng, n):
    """solve_equation.generate_equation_and_solution: an expression of length n - 2 (no multiplication), a digit
    replaced by x, '=' and the value; the answer is the digit."""
    n = max(n, 3)
    expr, val = expression(rng, n - 2, mult=False)
    idx = rng.randrange(len(expr))
    while expr[idx] not in "01234":
        idx = (idx + 1) % (n - 2)
    return list(f"{expr[:idx]}x{expr[idx + 1:]}={val}"), expr[idx]


# ------------------------------------------------ context-sensitive
def missing_duplicate(rng, n):
    n = max(n, 2)
    s = [rng.randrange(2) for _ in range(n // 2)]
    d = s + s
    i = rng.randrange(len(d))
    out = d[i]
    d[i] = 2
    return [str(x) for x in d] + ["3"] * (n % 2), str(out)


TASKS = {
    "even_pairs": ("R", even_pairs, 100.0, 96.4),
    "modular_arithmetic_simple": ("R", modular_arithmetic_simple, 100.0, 24.2),
    "parity_check": ("R", parity_check, 100.0, 52.0),
    "cycle_navigation": ("R", cycle_navigation, 100.0, 61.9),
    "modular_arithmetic": ("DCF", modular_arithmetic_brackets, 59.2, 32.5),
    "solve_equation": ("DCF", solve_equation, 67.8, 25.7),
    "missing_duplicate": ("CS", missing_duplicate, 54.3, 56.4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--teach", type=int, default=300)
    ap.add_argument("--longest", type=int, default=10)
    ap.add_argument("--asks", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in a.tasks:
        level, gen, lstm, transformer = TASKS[name]
        rng = random.Random(a.seed)
        draw = lambda lo, hi: [(tuple(s), y) for s, y in (gen(rng, rng.randint(lo, hi)) for _ in range(1))][0]
        data = [draw(1, a.longest) for _ in range(a.teach)]
        pool = {}
        for _ in range(2000):
            s, y = draw(1, 2 * a.longest)
            pool[s] = y
        syms = sorted({x for s, _ in data for x in s} | {x for s in pool for x in s})
        test = [draw(41, 500) for _ in range(400)]
        print(f"{name} ({level}): {len(syms)} symbols; taught on {a.teach} stories of 1-{a.longest}; may ask {a.asks} "
              f"of {len(pool)} of 1-{2 * a.longest}; tested on {len(test)} of 41-500. Published (best of 10 seeds, "
              f"taught on 1-40): LSTM {lstm}, transformer {transformer}", flush=True)
        t0 = time.time()
        m, asked = cells.learn(syms, lambda w: pool.get(tuple(w)), data, rng, a.asks, a.seconds, a.longest,
                               pool=list(pool), workers=a.workers)
        if m is None:
            print(f"  {name}: nothing found ({time.time() - t0:.0f} s)\n", flush=True)
            continue
        acc = 100 * sum(all(x in syms for x in s) and m.run(s) == y for s, y in test) / len(test)
        print(f"  {name}: {m}; asked {asked}; on 41-500: {acc:.1f} %  ({time.time() - t0:.0f} s)\n", flush=True)


if __name__ == "__main__":
    main()
