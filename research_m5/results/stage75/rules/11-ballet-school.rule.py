ALPHABET = ['plie', 'releve', 'turn', 'jump']


def answer(seq):
    shown = 'mirror'  # what the display shows
    p0 = 0  # a whole number
    for tok in seq:
        if tok == 'plie':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = 0
            p0 = n0
            shown = new
        elif tok == 'releve':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 - 1
            p0 = n0
            shown = new
        elif tok == 'turn':
            effect = {('mirror', 0): 'window', ('barre', 0): 'piano', ('piano', 0): 'mirror', ('window', 0): 'barre'}.get((shown, (p0 > 0) - (p0 < 0) + 1,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'jump':
            effect = {('mirror', 1): 'barre', ('barre', 1): 'mirror', ('window', 1): 'piano', ('piano', 1): 'window'}.get((shown, (p0 > 0) - (p0 < 0) + 1,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = 0
            p0 = n0
            shown = new
    return str(shown)
