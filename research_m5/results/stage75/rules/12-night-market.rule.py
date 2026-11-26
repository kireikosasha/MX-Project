ALPHABET = ['steam', 'buy', 'rain', 'clear']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'steam':
            effect = {(0, 1): 0}.get((0, int(p0 >= 3),), 1)
            new = shown + effect
            seen = effect
            d0, s0 = 0, None
            if seen == 1: d0 += 1
            n0 = s0 if s0 is not None else max(0, min(10 ** 9, p0 + d0))
            p0 = n0
            shown = new
        elif tok == 'buy':
            effect = {(1, 0): -1, (2, 0): -1}.get((min(max(shown, 0), 2), int(p1 >= 1),), 0)
            new = shown + effect
            seen = effect
            d0, s0 = 0, None
            if seen == -1: d0 += -1
            n0 = s0 if s0 is not None else max(0, min(10 ** 9, p0 + d0))
            p0 = n0
            shown = new
        elif tok == 'rain':
            effect = 0
            new = shown + effect
            seen = effect
            n1 = p1 + 1
            p1 = n1
            shown = new
        elif tok == 'clear':
            effect = 0
            new = shown + effect
            seen = effect
            n1 = 0
            p1 = n1
            shown = new
    return str(shown)
