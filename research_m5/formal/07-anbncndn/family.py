"""a^n b^n c^n d^n (n at least 1; Bhattamishra, Ahuja, Goyal 2020). "yes" when the story is 4 runs of the same length in this
order, "maybe" while it can still become one, "no" when it cannot."""
ALPHABET = ['a', 'b', 'c', 'd']
def answer(seq):
    runs, last = [], None
    for t in seq:
        i = ALPHABET.index(t)
        if last is None:
            if i != 0:
                return "no"
            runs.append(1)
        elif i == last:
            runs[-1] += 1
        elif i == last + 1:
            runs.append(1)
        else:
            return "no"
        last = i
        if len(runs) > 1 and runs[-1] > runs[0]:
            return "no"
    if not runs:
        return "maybe"
    if len(runs) == len(ALPHABET) and runs[-1] == runs[0] and all(r == runs[0] for r in runs[1:]):
        return "yes"
    if any(r != runs[0] for r in runs[1:-1]):
        return "no"
    return "maybe"
