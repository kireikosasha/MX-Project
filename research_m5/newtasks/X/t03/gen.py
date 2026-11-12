#!/usr/bin/env python3
"""Synthetic record-log task: a points game played in rounds.

Usage: python gen.py N SEED [SIZE]     (SIZE in 1, 3, 10; default 1)

Each episode is a log of records over players p1..p8:
  sc P A.   player P scores A points
  pn P A.   player P loses A points
  rd R.     round R ends (rounds are numbered 1, 2, ... in order)
Queries are answered from the simulated log above them:
  pt P R?   points of P when round R had just ended   (number)
  ahd P Q?  is P strictly ahead of Q right now         (yes / no)
  rk R?     player holding rank R right now, if unique (entity code)
"""
import random
import sys

PLAYERS = [f"p{i}" for i in range(1, 9)]


def pick_query(rng, players, pts, round_snaps):
    """Return (query_text_with_answer) valid at the current state, or None."""
    rounds_done = len(round_snaps)
    for _ in range(30):
        op = rng.choice(["pt", "ahd", "rk"])
        if op == "pt":
            if rounds_done == 0:
                continue
            p = rng.choice(players)
            n = rng.randint(1, rounds_done)
            return f"pt {p} {n}?\t{round_snaps[n - 1][p]}"
        if op == "ahd":
            x, y = rng.sample(players, 2)
            ans = "yes" if pts[x] > pts[y] else "no"
            return f"ahd {x} {y}?\t{ans}"
        order = sorted(players, key=lambda p: -pts[p])
        vals = [pts[p] for p in order]
        r = rng.randint(1, len(players))
        v = vals[r - 1]
        if (r >= 2 and vals[r - 2] == v) or (r < len(players) and vals[r] == v):
            continue  # tie at this rank: answer would be ambiguous
        return f"rk {r}?\t{order[r - 1]}"
    return None


def make_episode(rng, size, seen):
    """Build one episode as a list of row strings (without line numbers)."""
    while True:
        k = rng.randint(2, 4)
        players = rng.sample(PLAYERS, k)
        pts = {p: 0 for p in players}
        round_snaps = []
        records = []  # (row_text, points_after, rounds_done_after)
        n_rec = rng.randint(3, 12) * size
        for _ in range(n_rec):
            r = rng.random()
            if r < 0.2:
                round_snaps.append(dict(pts))
                text = f"rd {len(round_snaps)}."
            else:
                owing = [p for p in players if pts[p] > 0]
                if r < 0.3 and owing:
                    p = rng.choice(owing)
                    amt = rng.randint(1, min(pts[p], 3 * size))
                    pts[p] -= amt
                    text = f"pn {p} {amt}."
                else:
                    p = rng.choice(players)
                    amt = rng.randint(1, 5 * size)
                    pts[p] += amt
                    text = f"sc {p} {amt}."
            records.append((text, dict(pts), len(round_snaps)))

        q = rng.randint(1, 5)
        slots = sorted(rng.randint(1, len(records)) for _ in range(q))
        queries = {}
        for s in slots:
            _, st_pts, _ = records[s - 1]
            qa = pick_query(rng, players, st_pts, round_snaps[: records[s - 1][2]])
            if qa is not None:
                queries.setdefault(s, []).append(qa)
        if not queries:
            continue

        rows = []
        for i, (text, _, _) in enumerate(records, start=1):
            rows.append(text)
            rows.extend(queries.get(i, []))
        key = "\n".join(rows)
        if key in seen:
            continue
        seen.add(key)
        return rows


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n = int(argv[1])
    seed = int(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    seen = set()
    lines = []
    for _ in range(n):
        rows = make_episode(rng, size, seen)
        for i, row in enumerate(rows, start=1):
            lines.append(f"{i} {row}")
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
