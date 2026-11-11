#!/usr/bin/env python3
"""
The ledger: the two stages joined. Every number of a problem gets a role in a small table of facts, the
question gets a kind, and the answer is derived by code.

There are two accounts. A is what the question asks about, B is what it is compared or combined with.
The role of a number:
    +A -A *A /A     it adds to account A, takes from it, multiplies it, divides it (the same for B)
    >  <            a relation: A is this much more than B, A is this much less than B
    =S              the two accounts together are this much
    =A              account A ends up being this much (something of it is not told)
    0               it does not matter
An account is (what was added - what was taken) * multipliers / divisors. An account nothing is told about
directly is found from a relation or from the total: this is the solver part, and it is why "868 girls, 281
more girls than boys, how many pupils" needs no number twice - the state is used twice, not the number.
The kind of the question:
    val   how much is A            sum   A and B together        dif   how much more is one than the other
    pro   A times B                quo   A divided by B          mis   what is missing in A to be "=A"

A reading (the roles and the kind) is accepted only when it is well formed: no account without an amount,
nothing told that the answer does not need, one way to write each meaning. All accepted readings of a
problem are listed and computed; the network only says how likely each role and each kind is.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product

import numpy as np

ROLES = ["0", "+A", "-A", "*A", "/A", "+B", "-B", "*B", "/B", ">", "<", "=S", "=A"]
QUERIES = ["val", "sum", "dif", "pro", "quo", "mis"]
R = {r: i for i, r in enumerate(ROLES)}
Q = {q: i for i, q in enumerate(QUERIES)}
MAX_ACTIVE = 3


@lru_cache(maxsize=None)
def readings(k):
    """all well formed readings of k numbers: roles (n, k) and kinds (n,)"""
    rows = []
    for m in range(2, min(k, MAX_ACTIVE) + 1):
        for where in combinations(range(k), m):
            for roles in product(range(1, len(ROLES)), repeat=m):
                row = [0] * k
                for i, r in zip(where, roles):
                    row[i] = r
                rows.append(row)
    T = np.array(rows, dtype=np.int8)

    def any_of(*names):
        return np.isin(T, [R[n] for n in names]).any(1)

    add_a, add_b = any_of("+A", "-A"), any_of("+B", "-B")
    mem_a, mem_b = any_of("+A", "-A", "*A", "/A"), any_of("+B", "-B", "*B", "/B")
    n_rel = np.isin(T, [R[">"], R["<"]]).sum(1)
    n_s, n_e = (T == R["=S"]).sum(1), (T == R["=A"]).sum(1)
    ok = (n_rel + n_s + n_e <= 1) & (add_a | ~mem_a) & (add_b | ~mem_b)
    plain, rel, tot, eq = (n_rel + n_s + n_e == 0), n_rel == 1, n_s == 1, n_e == 1
    member = np.isin(T, [R[x] for x in ("+A", "-A", "*A", "/A", "+B", "-B", "*B", "/B")])
    first = T[np.arange(len(T)), member.argmax(1)]                 # the first number that belongs to an account
    first_a = np.isin(first, [R[x] for x in ("+A", "-A", "*A", "/A")])
    both = add_a & add_b & plain
    kinds = {
        "val": (add_a & ~mem_b & plain) | (~mem_a & add_b & (rel | tot)),
        "sum": (both & first_a) | (add_a & ~mem_b & rel),
        "dif": (both & first_a) | (~mem_a & add_b & tot),
        "pro": both & first_a,
        "quo": both,
        "mis": add_a & ~mem_b & eq,
    }
    out_t, out_q = [], []
    for q, mask in kinds.items():
        keep = ok & mask
        out_t.append(T[keep])
        out_q.append(np.full(int(keep.sum()), Q[q], dtype=np.int8))
    return np.concatenate(out_t), np.concatenate(out_q)


def compute(k, values):
    """the answers of all readings of k numbers with these values -> (answers, which are possible)"""
    T, kind = readings(k)
    V = np.asarray(values, dtype=np.float64)[None, :]

    def total(*names):
        return sum(((T == R[n]) * V).sum(1) for n in names)

    def times(name):
        return np.where(T == R[name], V, 1.0).prod(1)

    has_a, has_b = np.isin(T, [R["+A"], R["-A"]]).any(1), np.isin(T, [R["+B"], R["-B"]]).any(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (total("+A") - total("-A")) * times("*A") / times("/A")
        b = (total("+B") - total("-B")) * times("*B") / times("/B")
        rel = total(">") - total("<")                                           # A = B + rel
        has_rel, has_s = np.isin(T, [R[">"], R["<"]]).any(1), (T == R["=S"]).any(1)
        a2 = np.where(has_a, a, np.where(has_rel, b + rel, total("=S") - b))    # an account that is not told is found
        b2 = np.where(has_b, b, a - rel)
        out = np.select([kind == Q["val"], kind == Q["sum"], kind == Q["dif"], kind == Q["pro"], kind == Q["quo"]],
                        [a2, a2 + b2, np.abs(a - b), a * b, a / b], np.abs(total("=A") - a))
        ok = np.isfinite(out) & (out > 1e-9)
        ok &= np.where(kind == Q["mis"], True, a2 > 1e-9)
        out = np.where((kind == Q["dif"]) & has_s, np.abs(a2 - b), out)
        ok = ok & np.isfinite(out) & (out > 1e-9)
        ok &= np.where(np.isin(kind, [Q["sum"], Q["dif"], Q["pro"], Q["quo"]]) | (kind == Q["val"]) & (has_rel | has_s), b2 > 1e-9, True)
    return out, ok


def run(roles, kind, nums):
    """one reading computed exactly (fractions); None when it is not possible"""
    def acc(g):
        s, m, has = Fraction(0), Fraction(1), False
        for r, v in zip(roles, nums):
            name = ROLES[r]
            if name == "+" + g:
                s, has = s + v, True
            elif name == "-" + g:
                s, has = s - v, True
            elif name == "*" + g:
                m *= v
            elif name == "/" + g:
                if v == 0:
                    return None, False
                m /= v
        return s * m, has

    def one(name):
        return sum((v for r, v in zip(roles, nums) if ROLES[r] == name), Fraction(0))

    (a, has_a), (b, has_b) = acc("A"), acc("B")
    if a is None or b is None:
        return None
    rel, has_rel = one(">") - one("<"), any(ROLES[r] in "><" for r in roles)
    if not has_a:
        a = b + rel if has_rel else one("=S") - b
    if not has_b:
        b = a - rel
    q = QUERIES[kind]
    if q == "val":
        out = a
    elif q == "sum":
        out = a + b
    elif q == "dif":
        out = abs(a - b)
    elif q == "pro":
        out = a * b
    elif q == "quo":
        out = a / b if b != 0 else None
    else:
        out = abs(one("=A") - a)
    return out if out is not None and out > 0 else None


def show(roles, kind, nums=None):
    parts = []
    for i, r in enumerate(roles):
        if r:
            v = f"n{i}" if nums is None else (str(nums[i].numerator) if nums[i].denominator == 1 else str(float(nums[i])))
            parts.append(f"{v}:{ROLES[r]}")
    return f"{QUERIES[kind]}[{' '.join(parts)}]"


if __name__ == "__main__":
    import random
    for k in range(2, 8):
        print(k, "numbers:", len(readings(k)[0]), "readings")
    rng = random.Random(1)
    bad = n = 0
    for _ in range(300):                                    # the fast computation against the exact one
        k = rng.randint(2, 4)
        nums = [Fraction(rng.randint(1, 60)) for _ in range(k)]
        T, kind = readings(k)
        out, ok = compute(k, [float(x) for x in nums])
        for j in rng.sample(range(len(T)), 40):
            exact = run(T[j].tolist(), int(kind[j]), nums)
            n += 1
            if ok[j] and (exact is None or abs(float(exact) - out[j]) > 1e-6):
                bad += 1
    print("checked", n, "readings against exact fractions; differ:", bad)
    nums = [Fraction(868), Fraction(281)]
    roles = [R["+A"], R[">"]]
    print("868 girls, 281 more girls than boys, pupils:", show(roles, Q["sum"], nums), "=", run(roles, Q["sum"], nums))
