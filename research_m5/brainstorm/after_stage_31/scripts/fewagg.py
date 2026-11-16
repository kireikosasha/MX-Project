import re, sys, glob
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0.0, 0, 0]))   # n -> cond -> [sum acc, tasks, sum coverage, sum acc on known, tasks at 95, kinds]
per = defaultdict(dict)
for path in sorted(glob.glob("fewx_*.txt")):
    for line in open(path):
        m = re.match(r"(\w+) n=(\d+) task (\d+): (.*)", line)
        if not m:
            continue
        n = int(m.group(2))
        conds = m.group(4).split(" | ")
        if any(c == "late" for c in conds):
            continue
        for c in conds:
            g = re.match(r"(\w+) (\d+)/(\d+) known (\d+) right (\d+) kinds (\d+)", c)
            label, r, t, known, kr, kinds = g.group(1), *map(int, g.groups()[1:])
            a = agg[n][label]
            a[0] += r / t; a[1] += 1; a[2] += known / t; a[3] += (kr / known if known else 0); a[4] += r / t >= 0.95; a[5] += kinds
            per[(m.group(1), int(m.group(3)), n)][label] = r / t
for n in sorted(agg):
    for label in ("own", "text", "oracle"):
        a = agg[n][label]
        print(f"n={n} {label:6s}: tasks {a[1]:2d}, mean {100 * a[0] / a[1]:5.1f}%, at 95+ {a[4]}, questions of a kind met {100 * a[2] / a[1]:5.1f}%, right on those {100 * a[3] / a[1]:5.1f}%, kinds of statements {a[5] / a[1]:.1f}")
