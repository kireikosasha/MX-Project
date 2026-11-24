ALPHABET = ['0', '1', '~', '&', '|']


def answer(seq):
    shown = 'maybe'  # what the display shows
    p0 = []  # a stack, the last put on top
    for tok in seq:
        if tok == '0':
            effect = {('yes', 0): 'no', ('maybe', 0): 'yes'}.get((shown, ({'&': 1, '|': 2}[p0[-1]] if p0 else 0),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0[:-1]
            p0 = n0
            shown = new
        elif tok == '1':
            effect = {('maybe', 0): 'yes', ('yes', 0): 'no'}.get((shown, ({'&': 1, '|': 2}[p0[-1]] if p0 else 0),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0[:-1]
            p0 = n0
            shown = new
        elif tok == '~':
            effect = {('yes',): 'no'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == '&':
            effect = {('yes',): 'no'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + ['&']
            p0 = n0
            shown = new
        elif tok == '|':
            effect = {('yes',): 'no'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + ['|']
            p0 = n0
            shown = new
    return str(shown)
