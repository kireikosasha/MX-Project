ALPHABET = ['a', 'b']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'a':
            effect = {('yes', 0): 'maybe', ('maybe', 1): 'no'}.get((shown, int(p0 >= 3),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 0: d0 += 1
            n0 = s0 if s0 is not None else max(0, min(10 ** 9, p0 + d0))
            p0 = n0
            shown = new
        elif tok == 'b':
            effect = {('yes', 0): 'no', ('maybe', 0): 'yes'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = max(0, p0 - 1)
            p0 = n0
            shown = new
    return str(shown)
