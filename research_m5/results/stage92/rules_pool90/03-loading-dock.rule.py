ALPHABET = ['truck', 'crate', 'go']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a whole number
    p2 = 0  # a circle of 12
    p3 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'truck':
            effect = {(0, 1, 1): 1}.get((0, int(p0 >= 3), int(p1 >= 1),), 0)
            new = shown + effect
            seen = effect
            n0 = max(0, p0 - 3)
            n1 = p1 - 3
            n2 = (p2 - 3) % 12
            d3, s3 = 0, None
            if seen == 0: d3 += 1
            n3 = s3 if s3 is not None else max(0, min(10 ** 9, p3 + d3))
            p0 = n0
            p1 = n1
            p2 = n2
            p3 = n3
            shown = new
        elif tok == 'crate':
            effect = {(0, 11, 0): 1, (1, 11, 1): 1, (2, 3, 1): 1, (1, 11, 0): 1, (2, 5, 1): 1, (2, 11, 1): 1, (0, 5, 1): 1, (2, 2, 1): 1, (0, 2, 1): 1, (1, 5, 1): 1, (2, 7, 1): 1}.get((min(max(shown, 0), 2), p2, int(p3 >= 3),), 0)
            new = shown + effect
            seen = effect
            d0, s0 = 0, None
            if seen == 0: d0 += 1
            n0 = s0 if s0 is not None else max(0, min(10 ** 9, p0 + d0))
            n1 = p1 + 1
            n2 = (p2 + 1) % 12
            d3, s3 = 0, None
            if seen == 1: d3 += -2
            n3 = s3 if s3 is not None else max(0, min(10 ** 9, p3 + d3))
            p0 = n0
            p1 = n1
            p2 = n2
            p3 = n3
            shown = new
        elif tok == 'go':
            effect = 0
            new = shown + effect
            seen = effect
            n2 = 0
            p2 = n2
            shown = new
    return str(shown)
