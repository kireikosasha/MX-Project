ALPHABET = ['0', '1']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a level from 0 to 4
    for tok in seq:
        if tok == '0':
            effect = {('yes', 1): 'no'}.get((shown, int(p0 == 4),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 - 2 if 0 <= p0 - 2 <= 4 else p0
            p0 = n0
            shown = new
        elif tok == '1':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 3 if 0 <= p0 + 3 <= 4 else p0
            p0 = n0
            shown = new
    return str(shown)
