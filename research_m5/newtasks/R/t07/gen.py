import random
import sys

# Ledger task: holders h1..h5 hold amounts. Records set, add, remove and move
# amounts. Queries ask the current amount of a holder, which holder holds the
# most (rank 1), or whether one holder holds more than another.

HOLDERS = ["h1", "h2", "h3", "h4", "h5"]

R_SET = "zad"
R_ADD = "pim"
R_REMOVE = "kov"
R_MOVE = "lew"

Q_AMOUNT = "amt"
Q_TOP = "tpx"
Q_GREATER = "gtr"


def make_record(rng, size, amounts, touches):
    max_val = 9 * size
    op = rng.choices([R_SET, R_ADD, R_REMOVE, R_MOVE], weights=[2, 3, 2, 3])[0]
    has_money = any(amounts[h] > 0 for h in HOLDERS)
    if op in (R_REMOVE, R_MOVE) and not has_money:
        op = R_ADD

    if op == R_SET:
        h = rng.choice(HOLDERS)
        v = rng.randint(0, max_val)
        amounts[h] = v
        touches[h] += 1
        return f"{R_SET} {h} {v}."

    if op == R_ADD:
        h = rng.choice(HOLDERS)
        v = rng.randint(1, max_val)
        amounts[h] += v
        touches[h] += 1
        return f"{R_ADD} {h} {v}."

    if op == R_REMOVE:
        h = rng.choice([x for x in HOLDERS if amounts[x] > 0])
        v = rng.randint(1, amounts[h])
        amounts[h] -= v
        touches[h] += 1
        return f"{R_REMOVE} {h} {v}."

    src = rng.choice([x for x in HOLDERS if amounts[x] > 0])
    dst = rng.choice([x for x in HOLDERS if x != src])
    v = rng.randint(1, amounts[src])
    amounts[src] -= v
    amounts[dst] += v
    touches[src] += 1
    touches[dst] += 1
    return f"{R_MOVE} {src} {dst} {v}."


def make_query(rng, amounts, touches):
    kinds = [Q_AMOUNT, Q_TOP, Q_GREATER]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == Q_AMOUNT:
            cands = [h for h in HOLDERS if touches[h] >= 2]
            if not cands:
                continue
            h = rng.choice(cands)
            return f"{Q_AMOUNT} {h}?", str(amounts[h])

        if kind == Q_TOP:
            best = max(amounts.values())
            tops = [h for h in HOLDERS if amounts[h] == best]
            if best == 0 or len(tops) != 1 or touches[tops[0]] < 2:
                continue
            return f"{Q_TOP} 1?", tops[0]

        a, b = rng.sample(HOLDERS, 2)
        if amounts[a] == amounts[b] or touches[a] + touches[b] < 3:
            continue
        return f"{Q_GREATER} {a} {b}?", ("yes" if amounts[a] > amounts[b] else "no")
    return None


def try_episode(rng, size):
    n_rec = rng.randint(3 * size, 12 * size)
    n_q = rng.randint(1, 5)
    total = n_rec + n_q
    q_slots = set(rng.sample(range(1, total), n_q))

    amounts = {h: 0 for h in HOLDERS}
    touches = {h: 0 for h in HOLDERS}
    rows = []
    for i in range(total):
        if i in q_slots:
            made = make_query(rng, amounts, touches)
            if made is None:
                return None
            text, answer = made
            rows.append(f"{i + 1} {text}\t{answer}")
        else:
            rows.append(f"{i + 1} {make_record(rng, size, amounts, touches)}")
    return rows


def episode(rng, size):
    while True:
        rows = try_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1
    seed = int(argv[2]) if len(argv) > 2 else 0
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        raise SystemExit("SIZE must be 1, 3 or 10")

    rng = random.Random(seed)
    seen = set()
    out = []
    for _ in range(n):
        while True:
            rows = episode(rng, size)
            key = "\n".join(rows)
            if key not in seen:
                break
        seen.add(key)
        out.extend(rows)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv)
