import random
import sys

NAMES = ["ada", "bruno", "cara", "dev", "emil", "fern"]
ITEMS = ["lamp", "coin", "map", "rope", "bell", "cup"]


def question_for(rng, people, snap):
    cur, t_items, t_people = snap
    holders = [p for p in t_people if any(cur[i] == p for i in cur)]
    options = []
    if t_items:
        options += ["holder", "yesno"]
    if holders:
        options += ["inventory", "count"]
    if not options:
        return None
    kind = rng.choice(options)
    if kind == "holder":
        it = rng.choice(t_items)
        pat = rng.choice(["which one carries {}?", "which one holds {}?"])
        return pat.format(it) + "\t" + cur[it]
    if kind == "yesno":
        it = rng.choice(t_items)
        p = rng.choice(people)
        pat = rng.choice(["does {} carry {}?", "does {} hold {}?"])
        ans = "yes" if cur[it] == p else "no"
        return pat.format(p, it) + "\t" + ans
    p = rng.choice(holders)
    held = sorted(i for i in cur if cur[i] == p)
    if kind == "inventory":
        return "what does " + p + " carry?\t" + ",".join(held)
    return "how many items does " + p + " carry?\t" + str(len(held))


def make_story(rng):
    while True:
        people = rng.sample(NAMES, rng.randint(2, 4))
        items = rng.sample(ITEMS, rng.randint(2, 4))
        start = {it: rng.choice(people) for it in items}
        init = []
        for it in rng.sample(items, len(items)):
            verb = rng.choice(["carries", "holds"])
            init.append(start[it] + " " + verb + " " + it + ".")

        cur = dict(start)
        texts = []
        snaps = [(dict(cur), [], [])]
        t_items, t_people = [], []
        for _ in range(rng.randint(1, 6)):
            opts = [("pass", it, q) for it in items for q in people if q != cur[it]]
            opts += [("swap", a, b) for a in items for b in items
                     if a != b and cur[a] != cur[b]]
            op = rng.choice(opts)
            if op[0] == "pass":
                _, it, q = op
                p = cur[it]
                texts.append(p + " passes " + it + " to " + q + ".")
                cur[it] = q
                touched_i, touched_p = [it], [p, q]
            else:
                _, a, b = op
                x, y = cur[a], cur[b]
                texts.append(x + " swaps " + a + " for " + b + " with " + y + ".")
                cur[a], cur[b] = y, x
                touched_i, touched_p = [a, b], [x, y]
            for i in touched_i:
                if i not in t_items:
                    t_items.append(i)
            for p in touched_p:
                if p not in t_people:
                    t_people.append(p)
            snaps.append((dict(cur), sorted(t_items), sorted(t_people)))

        num = len(texts)
        placed = [[] for _ in range(num + 1)]
        for _ in range(rng.randint(1, 5)):
            pos = rng.randint(0, num)
            q = question_for(rng, people, snaps[pos])
            if q is not None:
                placed[pos].append(q)
        if not any(placed):
            continue

        lines = list(init)
        for k in range(num + 1):
            lines.extend(placed[k])
            if k < num:
                lines.append(texts[k])
        return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        for n, line in enumerate(make_story(rng), start=1):
            out.append(str(n) + " " + line)
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
