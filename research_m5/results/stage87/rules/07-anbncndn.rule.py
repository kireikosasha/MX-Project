ALPHABET = ['a', 'b', 'c', 'd']


def answer(seq):
    shown = 'maybe'  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'a':
            effect = {('maybe', 1): 'no'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n1 = p1 + 1
            p1 = n1
            shown = new
        elif tok == 'b':
            effect = {('maybe', 0): 'no'}.get((shown, int(p1 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 1
            n1 = max(0, p1 - 1)
            p0 = n0
            p1 = n1
            shown = new
        elif tok == 'c':
            effect = {('maybe', 0): 'no'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'd':
            effect = {('no',): None}.get((shown,), 'no')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
