#!/usr/bin/env python3
"""Synthetic record-log reasoning task: changing states.

Usage: python gen.py N SEED [SIZE]     (SIZE is 1, 3 or 10; default 1)

Objects (kind k) are created with a status (kind g) and a zone (kind h).
Records change them: set a status, set a zone, copy another object's status,
or swap zones of two objects. Queries ask the present status or zone of one
object, or which objects are currently in a given status or zone. Every answer
comes from simulating the records above the query.
"""
import random
import sys

OBJECTS = ["k1", "k2", "k3", "k4", "k5", "k6"]
STATUSES = ["g1", "g2", "g3"]
ZONES = ["h1", "h2", "h3"]


def make_record(rng, state, i):
    """Apply one random valid record (ordinal i) to state; return its text or None."""
    created = [o for o in OBJECTS if o in state]
    fresh = [o for o in OBJECTS if o not in state]
    status_pairs = [(a, b) for a in created for b in created
                    if a != b and state[a]["status"][0] != state[b]["status"][0]]
    zone_pairs = [(a, b) for a in created for b in created
                  if a < b and state[a]["zone"][0] != state[b]["zone"][0]]

    options, weights = [], []
    if fresh:
        options.append("mk")
        weights.append(4 if len(created) < 2 else 1)
    if created:
        options += ["vbe", "zmu"]
        weights += [3, 3]
    if status_pairs:
        options.append("cpy")
        weights.append(3)
    if zone_pairs:
        options.append("swz")
        weights.append(2)
    if not options:
        return None
    op = rng.choices(options, weights)[0]

    if op == "mk":
        o = rng.choice(fresh)
        s = rng.choice(STATUSES)
        z = rng.choice(ZONES)
        state[o] = {"status": (s, frozenset([i])), "zone": (z, frozenset([i]))}
        return f"mk {o} {s} {z}."
    if op == "vbe":
        o = rng.choice(created)
        cur = state[o]["status"][0]
        s = rng.choice([v for v in STATUSES if v != cur])
        state[o]["status"] = (s, frozenset([i]))
        return f"vbe {o} {s}."
    if op == "zmu":
        o = rng.choice(created)
        cur = state[o]["zone"][0]
        z = rng.choice([v for v in ZONES if v != cur])
        state[o]["zone"] = (z, frozenset([i]))
        return f"zmu {o} {z}."
    if op == "cpy":
        a, b = rng.choice(status_pairs)
        val, prov = state[b]["status"]
        state[a]["status"] = (val, prov | {i})
        return f"cpy {a} {b}."
    # swz
    a, b = rng.choice(zone_pairs)
    za, pa = state[a]["zone"]
    zb, pb = state[b]["zone"]
    state[a]["zone"] = (zb, pb | {i})
    state[b]["zone"] = (za, pa | {i})
    return f"swz {a} {b}."


def query_candidates(state):
    """Queries answerable now. A value counts only if at least two records
    shaped it (a copy or swap chain, or a set plus a later copy), and a
    'which' list needs at least two members, so no answer is a single lookup."""
    out = []
    names = sorted(state)
    for o in names:
        val, prov = state[o]["status"]
        if len(prov) >= 2:
            out.append((f"wst {o}?", val))
        val, prov = state[o]["zone"]
        if len(prov) >= 2:
            out.append((f"wlo {o}?", val))
    for g in STATUSES:
        members = [o for o in names if state[o]["status"][0] == g]
        if len(members) >= 2:
            out.append((f"whi {g}?", ",".join(members)))
    for h in ZONES:
        members = [o for o in names if state[o]["zone"][0] == h]
        if len(members) >= 2:
            out.append((f"whi {h}?", ",".join(members)))
    return out


def try_episode(rng, size):
    n_rec = rng.randint(3, 12) * size
    n_q = rng.randint(1, 5)
    state, recs, pool = {}, [], []
    for i in range(1, n_rec + 1):
        text = make_record(rng, state, i)
        if text is None:
            return None
        recs.append(text)
        for qt, ans in query_candidates(state):
            pool.append((i, qt, ans))
    rng.shuffle(pool)
    chosen, used = [], set()
    for i, qt, ans in pool:
        if len(chosen) == n_q:
            break
        if qt in used:
            continue
        used.add(qt)
        chosen.append((i, qt, ans))
    if len(chosen) < n_q:
        return None
    chosen.sort(key=lambda c: c[0])  # stable: keeps random order within a prefix
    rows, qi = [], 0
    for i in range(1, n_rec + 1):
        rows.append(recs[i - 1])
        while qi < len(chosen) and chosen[qi][0] == i:
            _, qt, ans = chosen[qi]
            rows.append(f"{qt}\t{ans}")
            qi += 1
    return rows


def make_episode(rng, size):
    for _ in range(1000):
        rows = try_episode(rng, size)
        if rows is not None:
            return rows
    raise RuntimeError("could not build an episode")


def main(argv):
    if len(argv) not in (3, 4):
        sys.stderr.write("usage: python gen.py N SEED [SIZE]\n")
        return 2
    n, seed = int(argv[1]), int(argv[2])
    size = int(argv[3]) if len(argv) == 4 else 1
    if size not in (1, 3, 10):
        sys.stderr.write("SIZE must be 1, 3 or 10\n")
        return 2
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        rows = make_episode(rng, size)
        for num, row in enumerate(rows, start=1):
            out.append(f"{num} {row}")
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
