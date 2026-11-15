"""Synthetic reading task: comparison on a scale (own wording, bAbI-style).

Usage: python gen.py N SEED

Prints N stories. Each story is a run of numbered lines starting at 1.
A statement compares two people on one scale ("Ann is taller than bob.").
A question ("Is ann taller than bob?" or "Who is the tallest?") is only
asked when the statements above it already settle the answer through a
chain of comparisons. Answers come from a hidden ranking of the people.
"""
import random
import sys

NAMES = ["ann", "bob", "carl", "dina", "eli", "fay", "gus", "hana"]

# (comparative, opposite comparative, superlative, opposite superlative)
SCALES = [
    ("taller", "shorter", "tallest", "shortest"),
    ("heavier", "lighter", "heaviest", "lightest"),
    ("older", "younger", "oldest", "youngest"),
    ("faster", "slower", "fastest", "slowest"),
]


def reaches(edges, src, dst):
    """True if src is above dst through a chain of statements (edges = (above, below))."""
    seen = {src}
    todo = [src]
    while todo:
        cur = todo.pop()
        for up, down in edges:
            if up == cur and down not in seen:
                if down == dst:
                    return True
                seen.add(down)
                todo.append(down)
    return False


def cap(text):
    return text[0].upper() + text[1:]


def ask(rng, mentioned, edges, height, more, less, most, least):
    """Pick a question that the statements above settle. Returns (text, answer) or None."""
    pairs = [(a, b) for a in mentioned for b in mentioned
             if a != b and (a, b) not in edges and reaches(edges, a, b)]
    sups = []
    if len(mentioned) >= 3:
        for x in mentioned:
            others = [y for y in mentioned if y != x]
            if all(reaches(edges, x, y) for y in others) and \
                    any((x, y) not in edges for y in others):
                sups.append((most, x))
            if all(reaches(edges, y, x) for y in others) and \
                    any((y, x) not in edges for y in others):
                sups.append((least, x))

    kinds = []
    if pairs:
        kinds.append("pair")
    if sups:
        kinds.append("sup")
    if not kinds:
        return None

    if rng.choice(kinds) == "pair":
        hi, lo = rng.choice(pairs)
        x, y = rng.choice([(hi, lo), (lo, hi)])
        word = rng.choice([more, less])
        truth = height[x] > height[y] if word == more else height[x] < height[y]
        answer = "yes" if truth else "no"
        return cap(f"Is {x} {word} than {y}?"), answer

    word, x = rng.choice(sups)
    return cap(f"Who is the {word}?"), x


def make_story(rng):
    """Build one story as a list of (line_text, answer_or_None). Returns None on failure."""
    more, less, most, least = rng.choice(SCALES)
    k = rng.randint(3, 5)
    people = rng.sample(NAMES, k)
    ranks = list(range(k))
    rng.shuffle(ranks)
    height = dict(zip(people, ranks))  # higher number = higher on the scale

    n_stmt = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    edges = set()
    mentioned = []
    lines = []
    n_s = 0
    n_q_done = 0

    while n_s < n_stmt or n_q_done < n_q:
        force_q = n_s >= n_stmt
        if n_q_done < n_q and n_s >= 1 and (force_q or rng.random() < 0.4):
            q = ask(rng, mentioned, edges, height, more, less, most, least)
            if q is not None:
                lines.append(q)
                n_q_done += 1
                continue
            if force_q:
                return None

        a, b = rng.sample(people, 2)
        if height[a] < height[b]:
            a, b = b, a  # a is the higher one
        if rng.random() < 0.5:
            text = f"{a} is {more} than {b}."
        else:
            text = f"{b} is {less} than {a}."
        edges.add((a, b))
        for p in (a, b):
            if p not in mentioned:
                mentioned.append(p)
        lines.append((cap(text), None))
        n_s += 1
    return lines


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        story = None
        while story is None:
            story = make_story(rng)
        for i, (text, answer) in enumerate(story, 1):
            if answer is None:
                print(f"{i} {text}")
            else:
                print(f"{i} {text}\t{answer}")


if __name__ == "__main__":
    main()
