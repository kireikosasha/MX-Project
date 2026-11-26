ALPHABET = ['back', 'flat', 'take', 'van']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'back':
            effect = 0
            new = shown + effect
            seen = effect
            n1 = p1 + 1
            p1 = n1
            shown = new
        elif tok == 'flat':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = p0 + 1
            p0 = n0
            shown = new
        elif tok == 'take':
            effect = {(0, 1): 0}.get((0, int(p1 >= 1),), 1)
            new = shown + effect
            seen = effect
            n1 = max(0, p1 - 1)
            p1 = n1
            shown = new
        elif tok == 'van':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = 0
            d1, s1 = 0, None
            d1 += 1 * p0
            n1 = s1 if s1 is not None else max(0, min(10 ** 9, p1 + d1))
            p0 = n0
            p1 = n1
            shown = new
    return str(shown)
