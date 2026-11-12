import random
import sys

# Chains of one relation: "zor X Y" means X now reports to Y (replacing any
# earlier link of X); "xul X" removes the upward link of X. Each entity has at
# most one link upward. Queries follow the relation two or more steps.

ENT = ["m%d" % i for i in range(1, 9)]  # one kind, 8 codes, same at every SIZE

SET_OP = "zor"
CUT_OP = "xul"
VUP_OP = "vup"    # who is k steps above x
TOP_OP = "qor"    # who is at the top of the chain of x
ABOVE_OP = "jiv"  # is y above x (yes / no)


def ancestors(x, boss):
    out = []
    while x in boss:
        x = boss[x]
        out.append(x)
    return out


def make_record(rng, boss):
    if boss and rng.random() < 0.2:
        x = rng.choice(sorted(boss))
        del boss[x]
        return ("%s %s." % (CUT_OP, x), None)
    pairs = []
    for x in ENT:
        for y in ENT:
            if y == x or boss.get(x) == y:
                continue
            if x in ancestors(y, boss):  # would close a cycle
                continue
            pairs.append((x, y))
    x, y = rng.choice(pairs)
    boss[x] = y
    return ("%s %s %s." % (SET_OP, x, y), None)


def make_query(rng, boss):
    cands = [x for x in ENT if len(ancestors(x, boss)) >= 2]
    if not cands:
        return None
    kind = rng.choice([VUP_OP, TOP_OP, ABOVE_OP])
    x = rng.choice(cands)
    anc = ancestors(x, boss)
    if kind == VUP_OP:
        k = rng.randint(2, len(anc))
        return ("%s %s %d?" % (VUP_OP, x, k), anc[k - 1])
    if kind == TOP_OP:
        return ("%s %s?" % (TOP_OP, x), anc[-1])
    others = [e for e in ENT if e != x and e not in anc]
    if others and rng.random() < 0.5:
        y = rng.choice(others)
        return ("%s %s %s?" % (ABOVE_OP, y, x), "no")
    y = rng.choice(anc[1:])
    return ("%s %s %s?" % (ABOVE_OP, y, x), "yes")


def make_episode(rng, size):
    while True:
        boss = {}
        n_rec = rng.randint(3 * size, 12 * size)
        n_q = rng.randint(1, 5)
        rows = []
        while n_rec > 0 or n_q > 0:
            if n_q > 0 and (n_rec == 0 or rng.random() < n_q / float(n_q + n_rec)):
                q = make_query(rng, boss)
                if q is not None:
                    rows.append(q)
                    n_q -= 1
                    continue
                if n_rec == 0:
                    break
            rows.append(make_record(rng, boss))
            n_rec -= 1
        if n_rec == 0 and n_q == 0:
            return rows


def render(rows):
    lines = []
    for i, (text, ans) in enumerate(rows, 1):
        if ans is None:
            lines.append("%d %s" % (i, text))
        else:
            lines.append("%d %s\t%s" % (i, text, ans))
    return lines


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        sys.exit(2)
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        sys.exit(2)
    rng = random.Random(seed)
    seen = set()
    for _ in range(n):
        while True:
            lines = render(make_episode(rng, size))
            key = "\n".join(lines)
            if key not in seen:
                break
        seen.add(key)
        for line in lines:
            print(line)


if __name__ == "__main__":
    main(sys.argv)
