ALPHABET = ['a', 'b']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a whole number
    for tok in seq:
        if tok == 'a':
            effect = {('yes', 0): 'maybe', ('maybe', 1): 'no'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 1
            n1 = p1 + 1
            p0 = n0
            p1 = n1
            shown = new
        elif tok == 'b':
            effect = {('yes', 1): 'no', ('maybe', 0): 'no', ('maybe', 1): 'yes'}.get((shown, (p1 > 0) - (p1 < 0) + 1,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = max(0, p0 - 1)
            d1, s1 = 0, None
            if seen == 0: d1 += -2
            n1 = s1 if s1 is not None else p1 + d1
            p0 = n0
            p1 = n1
            shown = new
    return str(shown)
