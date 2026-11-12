import random
import sys

# Library lending as records: persons p1..p8 borrow and return books b1..b8.
# Record ops: acq (book enters catalog), wdr (book leaves catalog),
#             zor (person borrows book), rvt (person returns book).
# Query ops:  hld (books held by a person, sorted list),
#             cnt (number of books held by a person),
#             avl (is a book in the catalog and on the shelf: yes/no).

PERSONS = ["p%d" % i for i in range(1, 9)]
BOOKS = ["b%d" % i for i in range(1, 9)]
REC_WEIGHTS = {"acq": 1, "wdr": 1, "zor": 4, "rvt": 2}


def mentions(history, code):
    return sum(1 for _, args in history if code in args)


def make_episode(rng, size):
    records_n = rng.randint(3 * size, 12 * size)
    queries_n = rng.randint(1, 5)
    total = records_n + queries_n
    qpos = set(rng.sample(range(3, total), queries_n))
    catalog = set()
    holder = {}  # book -> person currently holding it
    history = []  # (op, args) of records so far
    asked = set()
    rows = []
    for i in range(total):
        if i in qpos:
            cands = []
            for p in PERSONS:
                if mentions(history, p) >= 2:
                    held = sorted(b for b, h in holder.items() if h == p)
                    if held:
                        cands.append(("hld", p, ",".join(held)))
                    cands.append(("cnt", p, str(len(held))))
            for b in BOOKS:
                if mentions(history, b) >= 2:
                    ans = "yes" if (b in catalog and b not in holder) else "no"
                    cands.append(("avl", b, ans))
            cands = [c for c in cands if (c[0], c[1]) not in asked]
            if not cands:
                return None
            choice = rng.choice(cands)
            asked.add((choice[0], choice[1]))
            rows.append(("q",) + choice)
        else:
            options = {}
            for b in BOOKS:
                if b not in catalog:
                    options.setdefault("acq", []).append((b,))
                elif b in holder:
                    options.setdefault("rvt", []).append((holder[b], b))
                else:
                    options.setdefault("wdr", []).append((b,))
                    for p in PERSONS:
                        options.setdefault("zor", []).append((p, b))
            ops = sorted(options)
            op = rng.choices(ops, weights=[REC_WEIGHTS[o] for o in ops])[0]
            args = rng.choice(options[op])
            if op == "acq":
                catalog.add(args[0])
            elif op == "wdr":
                catalog.discard(args[0])
            elif op == "zor":
                holder[args[1]] = args[0]
            else:
                del holder[args[1]]
            history.append((op, args))
            rows.append(("r", op, args))
    return rows


def render(rows):
    lines = []
    for idx, row in enumerate(rows, 1):
        if row[0] == "r":
            _, op, args = row
            lines.append("%d %s %s." % (idx, op, " ".join(args)))
        else:
            _, op, arg, ans = row
            lines.append("%d %s %s?\t%s" % (idx, op, arg, ans))
    return lines


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            rows = None
            while rows is None:
                rows = make_episode(rng, size)
            lines = render(rows)
            key = "\n".join(lines)
            if key not in seen:
                break
        seen.add(key)
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
