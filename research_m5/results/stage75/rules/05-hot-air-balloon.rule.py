ALPHABET = ['burn', 'vent', 'wait']


def answer(seq):
    shown = 'ground'  # what the display shows
    for tok in seq:
        if tok == 'burn':
            effect = {('ground',): 'low', ('mid',): 'high', ('high',): None}.get((shown,), 'mid')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'vent':
            effect = {('ground',): None, ('low',): 'ground', ('mid',): 'low'}.get((shown,), 'mid')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'wait':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
