ALPHABET = ['0', '1']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a circle of 2
    p1 = 0  # a circle of 2
    for tok in seq:
        if tok == '0':
            effect = {('no', 1): None, ('yes', 0): 'no'}.get((shown, int(p0 >= 1),), 'yes')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n1 = (p1 + 1) % 2
            p1 = n1
            shown = new
        elif tok == '1':
            effect = {('yes', 0): 'no', ('no', 1): None}.get((shown, int(p1 >= 1),), 'yes')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = (p0 + 1) % 2
            p0 = n0
            shown = new
    return str(shown)
