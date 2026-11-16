import random
import sys

# Individuals, kinds and traits of the small world.
NAMES = ["ada", "bo", "cyn", "dax", "eli", "fen", "gus", "hal"]
KINDS = ["bird", "fish", "lizard", "robin", "trout", "newt"]
TRAITS = ["wings", "fins", "scales", "claws"]


def cap(word):
    return word[:1].upper() + word[1:]


def chain_of(kind, parent_of):
    # the kind itself first, then its ancestors, nearest first
    chain = [kind]
    while chain[-1] in parent_of:
        chain.append(parent_of[chain[-1]])
    return chain


def make_question_options(member, parent_of, trait_of):
    options = []
    for who, kind in member.items():
        chain = chain_of(kind, parent_of)
        if len(chain) >= 2:
            text = "Which kinds does %s belong to?" % cap(who)
            options.append((text, ",".join(chain)))
        for trait in TRAITS:
            verdict = None
            # the nearest kind in the chain that speaks about the trait decides
            for c in chain:
                if (c, trait) in trait_of:
                    verdict = "yes" if trait_of[(c, trait)] else "no"
                    break
            if verdict is not None:
                options.append(("Does %s have %s?" % (cap(who), trait), verdict))
    return options


def try_story(rng):
    people = rng.sample(NAMES, rng.randint(2, 4))
    kinds = rng.sample(KINDS, rng.randint(3, 5))
    parent_of = {}
    for i, k in enumerate(kinds):
        if i > 0 and rng.random() < 0.6:
            parent_of[k] = rng.choice(kinds[:i])

    pool = [("member", p, rng.choice(kinds)) for p in people]
    pool += [("falls", k, p) for k, p in parent_of.items()]
    pairs = [(k, t) for k in kinds for t in TRAITS]
    pool += [("trait", k, t, rng.random() < 0.5)
             for k, t in rng.sample(pairs, rng.randint(3, 8))]
    rng.shuffle(pool)

    n_stmt = min(rng.randint(3, 12), len(pool))
    if n_stmt < 3:
        return None
    n_q = rng.randint(1, 5)

    member, known_parent, known_trait = {}, {}, {}
    asked, lines = set(), []
    stmts, qs = 0, 0
    while stmts < n_stmt or qs < n_q:
        if qs < n_q and (stmts >= n_stmt or rng.random() < 0.5):
            options = [o for o in make_question_options(member, known_parent, known_trait)
                       if o[0] not in asked]
            if options:
                text, answer = rng.choice(options)
                asked.add(text)
                qs += 1
                lines.append("%d %s\t%s" % (len(lines) + 1, text, answer))
                continue
        if stmts >= n_stmt:
            return None
        fact = pool.pop()
        if fact[0] == "member":
            member[fact[1]] = fact[2]
            text = "%s belongs to %s." % (cap(fact[1]), fact[2])
        elif fact[0] == "falls":
            known_parent[fact[1]] = fact[2]
            text = "%s falls under %s." % (cap(fact[1]), fact[2])
        else:
            _, k, t, has = fact
            known_trait[(k, t)] = has
            text = "%s %s %s." % (cap(k), "has" if has else "lacks", t)
        stmts += 1
        lines.append("%d %s" % (len(lines) + 1, text))
    return lines


def make_story(rng):
    while True:
        lines = try_story(rng)
        if lines:
            return lines


def main():
    count = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        out.extend(make_story(rng))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
