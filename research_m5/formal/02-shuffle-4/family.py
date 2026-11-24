"""Shuffle-4 (Bhattamishra, Ahuja, Goyal 2020): 4 kinds of brackets, each kind balanced by itself, the kinds
interleaved freely (() [] {} <>). "yes" when every kind is closed, "maybe" while none was closed before it was opened,
"no" from the first closing bracket with no open one of its kind."""
ALPHABET = ['(', ')', '[', ']', '{', '}', '<', '>']
PAIRS = ['()', '[]', '{}', '<>']
def answer(seq):
    depth = [0] * len(PAIRS)
    for t in seq:
        for i, p in enumerate(PAIRS):
            if t == p[0]:
                depth[i] += 1
            elif t == p[1]:
                depth[i] -= 1
                if depth[i] < 0:
                    return "no"
    return "yes" if not any(depth) else "maybe"
