ALPHABET = ['call', 'crew', 'storm', 'clear']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a whole number
    p1 = 0  # a counter, not below zero
    p2 = 0  # a level from 0 to 6
    p3 = 0  # a circle of 7
    for tok in seq:
        if tok == 'call':
            effect = {(0, 1, 0): 0}.get((0, int(p0 >= 1), int(p1 >= 1),), 1)
            new = shown + effect
            seen = effect
            n0 = p0 - 1
            d2, s2 = 0, None
            if seen == 0: s2 = 0
            n2 = s2 if s2 is not None else max(0, min(6, p2 + d2))
            p0 = n0
            p2 = n2
            shown = new
        elif tok == 'crew':
            effect = {(1, 0): -1, (2, 0): -1}.get((min(max(shown, 0), 2), int(p1 >= 1),), 0)
            new = shown + effect
            seen = effect
            n0 = p0 + 1
            d2, s2 = 0, None
            if seen == 0: d2 += 1
            n2 = s2 if s2 is not None else max(0, min(6, p2 + d2))
            d3, s3 = 0, None
            if seen == 0: d3 += 1
            n3 = s3 if s3 is not None else (p3 + d3) % 7
            p0 = n0
            p2 = n2
            p3 = n3
            shown = new
        elif tok == 'storm':
            effect = 0
            new = shown + effect
            seen = effect
            n1 = p1 + 1
            p1 = n1
            shown = new
        elif tok == 'clear':
            effect = {(1, 1, 0): -1, (2, 4, 0): -5, (2, 1, 0): -1, (2, 5, 0): -5, (2, 6, 1): -6, (1, 6, 0): -1, (2, 6, 0): -2, (1, 2, 0): -1, (1, 3, 0): -1, (2, 4, 1): -4}.get((min(max(shown, 0), 2), p2, int(p3 == 6),), 0)
            new = shown + effect
            seen = effect
            n1 = 0
            d2, s2 = 0, None
            if seen == -1: d2 += -1
            n2 = s2 if s2 is not None else max(0, min(6, p2 + d2))
            p1 = n1
            p2 = n2
            shown = new
    return str(shown)
