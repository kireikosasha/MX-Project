ALPHABET = ['ship', 'pilot', 'fog', 'clear']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a whole number
    p1 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'ship':
            effect = {(0, 1, 0): 0}.get((0, int(p0 >= 1), int(p1 >= 1),), 1)
            new = shown + effect
            seen = effect
            n0 = p0 - 1
            p0 = n0
            shown = new
        elif tok == 'pilot':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = p0 + 1
            p0 = n0
            shown = new
        elif tok == 'fog':
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
