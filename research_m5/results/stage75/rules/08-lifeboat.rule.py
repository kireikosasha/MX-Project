ALPHABET = ['wave', 'lull', 'swap']


def answer(seq):
    shown = 'dry'  # what the display shows
    p0 = 0  # a circle of 2
    for tok in seq:
        if tok == 'wave':
            effect = {('dry', 0): 'damp', ('damp', 0): 'wet', ('wet', 0): 'awash'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'lull':
            effect = {('damp', 1): 'dry', ('awash', 1): 'wet', ('wet', 1): 'damp'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'swap':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = (p0 + 1) % 2
            p0 = n0
            shown = new
    return str(shown)
