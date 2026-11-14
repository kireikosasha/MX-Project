import random
import sys

NAMES = ["ann", "bob", "cat", "dan", "eve", "fay", "gus", "hal"]
GROUPS = ["red", "blue", "green", "gold"]


def make_story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_st + n_q
    q_slots = set(rng.sample(range(1, total), n_q))
    member = {}  # name -> group; each person is in at most one group
    lines = []
    for slot in range(total):
        if slot in q_slots:
            kind = rng.randrange(4)
            if kind == 0:
                p = rng.choice(NAMES)
                q = "Which group is %s in?" % p.capitalize()
                ans = member.get(p, "none")
            elif kind == 1:
                g = rng.choice(GROUPS)
                q = "Who is in the %s group?" % g
                members = sorted(x for x in member if member[x] == g)
                ans = ",".join(members) if members else "none"
            elif kind == 2:
                g = rng.choice(GROUPS)
                q = "How many are in the %s group?" % g
                ans = str(sum(1 for x in member if member[x] == g))
            else:
                p = rng.choice(NAMES)
                g = rng.choice(GROUPS)
                q = "Is %s in the %s group?" % (p.capitalize(), g)
                ans = "yes" if member.get(p) == g else "no"
            lines.append(q + "\t" + ans)
        else:
            out = [x for x in NAMES if x not in member]
            inn = sorted(member)
            options = []
            if out:
                options.append("join")
            if inn:
                options.append("leave")
                options.append("move")
            act = rng.choice(options)
            if act == "join":
                p = rng.choice(out)
                g = rng.choice(GROUPS)
                member[p] = g
                s = "%s joins the %s group." % (p.capitalize(), g)
            elif act == "leave":
                p = rng.choice(inn)
                g = member.pop(p)
                s = "%s leaves the %s group." % (p.capitalize(), g)
            else:
                p = rng.choice(inn)
                g = rng.choice([x for x in GROUPS if x != member[p]])
                member[p] = g
                s = "%s moves to the %s group." % (p.capitalize(), g)
            lines.append(s)
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for i, text in enumerate(make_story(rng), 1):
            out.append("%d %s" % (i, text))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
