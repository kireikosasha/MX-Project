ALPHABET = ['sack', 'gust', 'brake', 'lull']


def answer(seq):
    shown = 'empty'  # what the display shows
    p0 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'sack':
            effect = {('low',): 'high', ('high',): 'full', ('full',): None}.get((shown,), 'low')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'gust':
            effect = {('low', 0): 'empty', ('high', 0): 'low', ('full', 0): 'high'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'brake':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 1
            p0 = n0
            shown = new
        elif tok == 'lull':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = 0
            p0 = n0
            shown = new
    return str(shown)
