#!/usr/bin/env python3
"""Synthetic kitchen-records reasoning task (dishes and ingredients).

Records:
  vek d1 i3.   dish d1 needs ingredient i3
  lom i3.      ingredient i3 is stocked
  zib i5.      ingredient i5 is banned from the kitchen
  quv d2.      dish d2 is taken off the menu

Queries (answered only from the rows above them):
  pex d1?      can dish d1 be made? (on menu, and every needed ingredient usable)
  gop d1?      which needed ingredients of dish d1 are not usable (sorted list)
  rud i3?      which dishes that need ingredient i3 can be made (sorted list)

An ingredient is usable when it is stocked and not banned.
Usage: python gen.py N SEED [SIZE]     SIZE is 1, 3 or 10 (default 1).
"""

import random
import sys

REC_WEIGHTS = [("vek", 5), ("lom", 3), ("zib", 2), ("quv", 2)]
MAX_CODE = 8


class Kitchen:
    def __init__(self):
        self.records = []   # (op, args) in order
        self.needs = {}     # dish -> set of ingredients
        self.stocked = set()
        self.banned = set()
        self.off = set()

    def add(self, op, args):
        self.records.append((op, args))
        if op == "vek":
            d, i = args
            self.needs.setdefault(d, set()).add(i)
        elif op == "lom":
            self.stocked.add(args[0])
        elif op == "zib":
            self.banned.add(args[0])
        elif op == "quv":
            self.off.add(args[0])

    def usable(self, i):
        return i in self.stocked and i not in self.banned

    def makeable(self, d):
        return (d in self.needs and d not in self.off
                and all(self.usable(i) for i in self.needs[d]))

    def missing(self, d):
        return sorted(i for i in self.needs.get(d, ()) if not self.usable(i))

    def dishes_using(self, i):
        return sorted(d for d, s in self.needs.items() if i in s)

    def support_dish(self, d):
        ings = self.needs.get(d, set())
        n = 0
        for op, args in self.records:
            if op == "vek" and args[0] == d:
                n += 1
            elif op in ("lom", "zib") and args[0] in ings:
                n += 1
            elif op == "quv" and args[0] == d:
                n += 1
        return n

    def support_ing(self, i):
        dishes = set(self.dishes_using(i))
        n = 0
        for op, args in self.records:
            if op == "vek" and args[1] == i:
                n += 1
            elif op in ("lom", "zib") and args[0] == i:
                n += 1
            elif op == "quv" and args[0] in dishes:
                n += 1
        return n

    def queries(self):
        """All valid (op, entity, answer) queries given the rows so far."""
        out = []
        for d in sorted(self.needs):
            if self.support_dish(d) < 2:
                continue
            out.append(("pex", d, "yes" if self.makeable(d) else "no"))
            miss = self.missing(d)
            if miss:
                out.append(("gop", d, ",".join(miss)))
        ingredients = sorted({i for s in self.needs.values() for i in s})
        for i in ingredients:
            ans = [d for d in self.dishes_using(i) if self.makeable(d)]
            if ans and self.support_ing(i) >= 2:
                out.append(("rud", i, ",".join(ans)))
        return out


def random_record(rng):
    ops = [op for op, _ in REC_WEIGHTS]
    weights = [w for _, w in REC_WEIGHTS]
    op = rng.choices(ops, weights=weights)[0]
    if op == "vek":
        args = (f"d{rng.randint(1, MAX_CODE)}", f"i{rng.randint(1, MAX_CODE)}")
    elif op in ("lom", "zib"):
        args = (f"i{rng.randint(1, MAX_CODE)}",)
    else:
        args = (f"d{rng.randint(1, MAX_CODE)}",)
    return op, args


def try_episode(rng, size):
    """Return a list of row bodies (without numbers), or None if it failed."""
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    kitchen = Kitchen()
    rows = []
    seen = set()
    rec_left, q_left = n_rec, n_q
    while rec_left or q_left:
        cands = kitchen.queries() if q_left else []
        if cands and (rec_left == 0 or rng.random() < 0.35):
            op, ent, ans = rng.choice(cands)
            rows.append(f"{op} {ent}?\t{ans}")
            q_left -= 1
        elif rec_left:
            op, args = random_record(rng)
            text = " ".join((op,) + args) + "."
            for _ in range(10):
                if text not in seen:
                    break
                op, args = random_record(rng)
                text = " ".join((op,) + args) + "."
            seen.add(text)
            kitchen.add(op, args)
            rows.append(text)
            rec_left -= 1
        else:
            return None
    return rows


def make_episode(rng, size):
    while True:
        rows = try_episode(rng, size)
        if rows is not None:
            return rows


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        for idx, row in enumerate(make_episode(rng, size), 1):
            lines.append(f"{idx} {row}")
    sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))


if __name__ == "__main__":
    main(sys.argv)
