ALPHABET = ['spin', 'reverse', 'board', 'alight']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a circle of 2
    p1 = 0  # a circle of 5
    p2 = 0  # a row of 5 flags, one for each value of p1
    for tok in seq:
        if tok == 'spin':
            effect = 0
            new = shown + effect
            seen = effect
            d1, s1 = 0, None
            if int(p0 >= 1) == 0: d1 += 1
            if int(p0 >= 1) == 1: d1 += -1
            n1 = s1 if s1 is not None else (p1 + d1) % 5
            p1 = n1
            shown = new
        elif tok == 'reverse':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = (p0 + 1) % 2
            p0 = n0
            shown = new
        elif tok == 'board':
            effect = {(0, 1): 0}.get((0, (p2 >> p1) & 1,), 1)
            new = shown + effect
            seen = effect
            n2 = p2 | (1 << p1)
            p2 = n2
            shown = new
        elif tok == 'alight':
            effect = {(0, 1): -1}.get((0, (p2 >> p1) & 1,), 0)
            new = shown + effect
            seen = effect
            n2 = p2 & ~ (1 << p1)
            p2 = n2
            shown = new
    return str(shown)
