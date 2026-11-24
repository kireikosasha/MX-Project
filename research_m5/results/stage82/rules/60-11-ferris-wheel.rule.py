ALPHABET = ['turn', 'board', 'exit']


def answer(seq):
    shown = 0  # what the display shows
    p0 = 0  # a circle of 4
    p1 = 0  # a row of 4 flags, one for each value of p0
    for tok in seq:
        if tok == 'turn':
            effect = 0
            new = shown + effect
            seen = effect
            n0 = (p0 + 1) % 4
            p0 = n0
            shown = new
        elif tok == 'board':
            effect = {(0, 1): 0}.get((0, (p1 >> p0) & 1,), 1)
            new = shown + effect
            seen = effect
            n1 = p1 | (1 << p0)
            p1 = n1
            shown = new
        elif tok == 'exit':
            effect = {(0, 1): -1}.get((0, (p1 >> p0) & 1,), 0)
            new = shown + effect
            seen = effect
            n1 = p1 & ~ (1 << p0)
            p1 = n1
            shown = new
    return str(shown)
