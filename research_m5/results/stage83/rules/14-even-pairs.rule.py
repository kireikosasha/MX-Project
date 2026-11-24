ALPHABET = ['a', 'b']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'a':
            effect = {('yes', 1): 'no', ('no', 1): 'yes'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = 0
            n1 = p1 + 1
            p0 = n0
            p1 = n1
            shown = new
        elif tok == 'b':
            effect = {('no', 1): 'yes', ('yes', 1): 'no'}.get((shown, int(p1 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 1
            n1 = 0
            p0 = n0
            p1 = n1
            shown = new
    return str(shown)
