ALPHABET = ['0', '1', '2']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a circle of 5
    for tok in seq:
        if tok == '0':
            effect = {(0,): 4}.get((min(max(shown, 0), 2),), -1)
            new = shown + effect
            seen = effect
            n0 = (p0 - 1) % 5
            p0 = n0
            shown = new
        elif tok == '1':
            effect = 0
            new = shown + effect
            seen = effect
            shown = new
        elif tok == '2':
            effect = {(0, 1): -4}.get((0, int(p0 == 4),), 1)
            new = shown + effect
            seen = effect
            n0 = (p0 + 1) % 5
            p0 = n0
            shown = new
    return str(shown)
