ALPHABET = ['0', '1']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a circle of 2
    for tok in seq:
        if tok == '0':
            effect = {('yes', 0, 1): 'no', ('no', 1, 0): 'yes', ('yes', 1, 0): 'no'}.get((shown, int(p0 >= 1), int(p1 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 1: d0 += 1
            n0 = s0 if s0 is not None else max(0, min(10 ** 9, p0 + d0))
            n1 = 0
            p0 = n0
            p1 = n1
            shown = new
        elif tok == '1':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = 0
            n1 = (p1 + 1) % 2
            p0 = n0
            p1 = n1
            shown = new
    return str(shown)
