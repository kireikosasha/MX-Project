ALPHABET = ['0', '1']


def answer(seq):
    shown = 'yes'  # what the display shows
    for tok in seq:
        if tok == '0':
            effect = {('no',): None}.get((shown,), 'no')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == '1':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
