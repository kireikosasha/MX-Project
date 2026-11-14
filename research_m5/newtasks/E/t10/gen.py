import random
import sys

# Vocabulary (about 26 words): 6 names, 4 group names, plus fixed pattern words.
NAMES = ["Ann", "Bob", "Cal", "Dee", "Eve", "Gus"]
GROUPS = ["chess", "choir", "crew", "band"]


def pick_event(rng, where):
    """List every legal statement given the current state, then pick one."""
    options = []
    for n in NAMES:
        if n not in where:
            for g in GROUPS:
                options.append(("join", n, g))
        else:
            cur = where[n]
            options.append(("leave", n, cur))
            for g in GROUPS:
                if g != cur:
                    options.append(("move", n, cur, g))
    return rng.choice(options)


def make_story(rng):
    where = {}                               # name -> group it is in
    members = {g: set() for g in GROUPS}     # group -> set of names
    mentioned = []                           # names said so far, in order
    asked = set()
    lines = []

    s_left = rng.randint(3, 12)              # statements
    q_left = rng.randint(1, 5)               # questions

    while s_left or q_left:
        num = len(lines) + 1
        ask = (q_left > 0 and bool(mentioned) and
               (s_left == 0 or rng.random() < q_left / (s_left + q_left)))
        if ask:
            cands = []
            for n in mentioned:
                cands.append(("which", n))
            for g in GROUPS:
                cands.append(("who", g))
                cands.append(("how", g))
            cands = [c for c in cands if c not in asked]
            kind, arg = rng.choice(cands)
            asked.add((kind, arg))
            if kind == "which":
                text = "Which group is %s in?" % arg
                answer = where.get(arg, "none")
            elif kind == "who":
                text = "Who is in the %s group?" % arg
                names = sorted(members[arg])
                answer = ",".join(x.lower() for x in names) if names else "none"
            else:
                text = "How many people are in the %s group?" % arg
                answer = str(len(members[arg]))
            lines.append("%d %s\t%s" % (num, text, answer))
            q_left -= 1
        else:
            ev = pick_event(rng, where)
            if ev[0] == "join":
                _, n, g = ev
                where[n] = g
                members[g].add(n)
                text = "%s joins the %s group." % (n, g)
                mentioned_name = n
            elif ev[0] == "leave":
                _, n, g = ev
                del where[n]
                members[g].discard(n)
                text = "%s leaves the %s group." % (n, g)
                mentioned_name = n
            else:
                _, n, old, new = ev
                members[old].discard(n)
                where[n] = new
                members[new].add(n)
                text = "%s moves from the %s group to the %s group." % (n, old, new)
                mentioned_name = n
            if mentioned_name not in mentioned:
                mentioned.append(mentioned_name)
            lines.append("%d %s" % (num, text))
            s_left -= 1
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
