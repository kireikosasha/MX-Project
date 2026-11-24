ALPHABET = ['0', '1']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a circle of 3
    for tok in seq:
        if tok == '0':
            effect = {(0, 0): 'no', (0, 1): None}.get((0, p0,), 'yes')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = (p0 + 1) % 3
            p0 = n0
            shown = new
        elif tok == '1':
            effect = {(0, 0): 'no', (0, 2): None}.get((0, p0,), 'yes')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = (p0 - 1) % 3
            p0 = n0
            shown = new
    return str(shown)
