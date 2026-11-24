ALPHABET = ['x', 'y', 'undo']


def answer(seq):
    shown = 'empty'  # what the display shows
    p0 = []  # a stack, the last put on top
    for tok in seq:
        if tok == 'x':
            effect = {('x',): None}.get((shown,), 'x')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + ['x']
            p0 = n0
            shown = new
        elif tok == 'y':
            effect = {('y',): None}.get((shown,), 'y')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + ['y']
            p0 = n0
            shown = new
        elif tok == 'undo':
            effect = {('y', 1): 'x', ('y', 0): 'empty', ('x', 2): 'y', ('x', 0): 'empty'}.get((shown, ({'x': 1, 'y': 2}[p0[-2]] if len(p0) > 1 else 0),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0[:-1]
            p0 = n0
            shown = new
    return str(shown)
