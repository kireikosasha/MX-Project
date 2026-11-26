ALPHABET = ['log', 'wait', 'pizza']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a level from 0 to 4
    for tok in seq:
        if tok == 'log':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = min(4, p0 + 1)
            p0 = n0
            shown = new
        elif tok == 'wait':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = max(0, p0 - 1)
            p0 = n0
            shown = new
        elif tok == 'pizza':
            effect = {(0, 2): 1, (0, 3): 1}.get((0, p0,), 0)
            new = shown + effect
            seen = effect
            n0 = max(0, p0 - 1)
            p0 = n0
            shown = new
    return str(shown)
