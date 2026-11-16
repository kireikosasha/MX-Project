import random
import sys

# Task: who holds which object after a sequence of gets, gives and drops.
# Answering needs the latest transfer chain, not a single sentence.

NAMES = ["Ann", "Ben", "Cal", "Dee", "Eve", "Fay"]
ITEMS = ["apple", "book", "cup", "hat", "ball", "pen", "coin"]


def make_statement(rng, holder):
    while True:
        kind = rng.choice(["get", "give", "drop"])
        it = rng.choice(ITEMS)
        p = rng.choice(NAMES)
        if kind == "get" and holder[it] is None:
            holder[it] = p
            return f"{p} gets the {it}.", it
        if kind == "give" and holder[it] == p:
            q = rng.choice([n for n in NAMES if n != p])
            holder[it] = q
            return f"{p} gives the {it} to {q}.", it
        if kind == "drop" and holder[it] == p:
            holder[it] = None
            return f"{p} drops the {it}.", it


def make_question(rng, holder, mentions):
    cands = [it for it in ITEMS if mentions[it] >= 2]
    if not cands:
        return None
    it = rng.choice(cands)
    if rng.random() < 0.5:
        ans = holder[it].lower() if holder[it] is not None else "nobody"
        return f"Who has the {it}?", ans
    p = rng.choice(NAMES)
    ans = "yes" if holder[it] == p else "no"
    return f"Does {p} have the {it}?", ans


def make_story(rng):
    holder = {it: None for it in ITEMS}
    mentions = {it: 0 for it in ITEMS}
    n_stmt = rng.randint(3, 12)
    q_left = rng.randint(1, 5)
    out = []
    n_q_done = 0
    stmt_count = 0
    while stmt_count < n_stmt or q_left > 0:
        remaining = n_stmt - stmt_count
        ask = stmt_count >= 1 and q_left > 0 and (
            remaining == 0 or rng.random() < q_left / (q_left + remaining))
        if ask:
            q_left -= 1
            res = make_question(rng, holder, mentions)
            if res is not None:
                text, ans = res
                out.append(f"{text}\t{ans}")
                n_q_done += 1
            continue
        text, it = make_statement(rng, holder)
        mentions[it] += 1
        out.append(text)
        stmt_count += 1
    if n_q_done == 0:
        return make_story(rng)
    return out


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for i, line in enumerate(make_story(rng), 1):
            print(f"{i} {line}")


if __name__ == "__main__":
    main()
