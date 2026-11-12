#!/usr/bin/env python3
"""Synthetic record-log task: staffing qualification.

Records describe certificates granted to persons, lapsed certificates, and
requirements that jobs have for certificates. Queries ask whether a person is
qualified for a job, which persons are qualified for a job, or how many
certificates a person holds. Answers come from a simulation of the records.

Usage: python gen.py N SEED [SIZE]
"""
import random
import sys

PERSONS = [f"p{i}" for i in range(1, 9)]
JOBS = [f"j{i}" for i in range(1, 9)]
CERTS = [f"c{i}" for i in range(1, 9)]

REC_WEIGHTS = {"cgr": 3, "lp": 1, "rq": 3, "dq": 1}


def episode(rng, size):
    held = {p: set() for p in PERSONS}   # certificates each person holds now
    reqs = {j: set() for j in JOBS}      # certificates each job requires now
    ev = {p: 0 for p in PERSONS}         # grant/lapse records per person
    cap = min(8, 2 + 2 * (size - 1))     # longest requirement list allowed
    target_recs = rng.randint(3, 12) * size
    n_queries = rng.randint(1, 5)
    rows = []
    asked = set()
    nrec = 0

    def do_record():
        ops = {
            "cgr": [(p, c) for p in PERSONS for c in CERTS if c not in held[p]],
            "lp": [(p, c) for p in PERSONS for c in sorted(held[p])],
            "rq": [(j, c) for j in JOBS if len(reqs[j]) < cap
                   for c in CERTS if c not in reqs[j]],
            "dq": [(j, c) for j in JOBS for c in sorted(reqs[j])],
        }
        avail = [o for o in REC_WEIGHTS if ops[o]]
        op = rng.choices(avail, [REC_WEIGHTS[o] for o in avail])[0]
        a, b = rng.choice(ops[op])
        if op == "cgr":
            held[a].add(b)
            ev[a] += 1
        elif op == "lp":
            held[a].discard(b)
            ev[a] += 1
        elif op == "rq":
            reqs[a].add(b)
        else:
            reqs[a].discard(b)
        rows.append(f"{op} {a} {b}.")

    def query_options():
        out = []
        for j in JOBS:
            if len(reqs[j]) >= 2:
                qual = [p for p in PERSONS if reqs[j] <= held[p]]
                if qual:
                    out.append((f"wq {j}?", ",".join(qual)))
            if not reqs[j]:
                continue
            for p in PERSONS:
                if ev[p] >= 1:
                    ans = "yes" if reqs[j] <= held[p] else "no"
                    out.append((f"qy {p} {j}?", ans))
        for p in PERSONS:
            if ev[p] >= 2:
                out.append((f"nc {p}?", str(len(held[p]))))
        return [(q, a) for q, a in out if q not in asked]

    qleft = n_queries
    while qleft > 0 or nrec < target_recs:
        opts = query_options() if qleft > 0 else []
        if nrec >= target_recs and qleft < n_queries and not opts:
            break  # record budget used and at least one query asked
        if opts and nrec >= 2 and (nrec >= target_recs or rng.random() < 0.35):
            q, a = rng.choice(opts)
            asked.add(q)
            rows.append(f"{q}\t{a}")
            qleft -= 1
        else:
            do_record()
            nrec += 1
    return rows


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python gen.py N SEED [SIZE]")
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if size not in (1, 3, 10):
        sys.exit("SIZE must be 1, 3 or 10")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        while True:  # redraw an episode that overruns the record cap
            rows = episode(rng, size)
            if sum(1 for r in rows if "\t" not in r) <= 12 * size:
                break
        for i, row in enumerate(rows, 1):
            out.append(f"{i} {row}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
