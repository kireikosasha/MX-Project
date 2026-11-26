ALPHABET = ['kid', 'honk', 'rain', 'refill']


def answer(seq):
    shown = 3  # what the display shows
    p0 = 0  # a circle of 2
    p1 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'kid':
            effect = {(2, 1): -1, (1, 1): -1}.get((min(max(shown, 0), 2), int(p0 >= 1),), 0)
            new = shown + effect
            seen = effect
            d1, s1 = 0, None
            if seen == -1: d1 += 1
            n1 = s1 if s1 is not None else max(0, min(10 ** 9, p1 + d1))
            p1 = n1
            shown = new
        elif tok == 'honk':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = (p0 + 1) % 2
            p0 = n0
            shown = new
        elif tok == 'rain':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = 0
            p0 = n0
            shown = new
        elif tok == 'refill':
            effect = p1
            new = shown + effect
            seen = effect
            n1 = 0
            p1 = n1
            shown = new
    return str(shown)
