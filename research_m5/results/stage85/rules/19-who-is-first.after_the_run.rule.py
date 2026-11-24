ALPHABET = ['a', 'b', 'serve']


def answer(seq):
    shown = 'nobody'  # what the display shows
    p0 = []  # a line, first come first served
    for tok in seq:
        if tok == 'a':
            effect = {('nobody',): 'a'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0
            if seen == 0: n0 = p0 + ['a']
            p0 = n0
            shown = new
        elif tok == 'b':
            effect = {('nobody',): 'b'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0
            if seen == 0: n0 = p0 + ['b']
            p0 = n0
            shown = new
        elif tok == 'serve':
            effect = {('b', 1): 'a', ('b', 0): 'nobody', ('a', 2): 'b', ('a', 0): 'nobody'}.get((shown, ({'a': 1, 'b': 2}[p0[0]] if p0 else 0),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0[1:]
            p0 = n0
            shown = new
    return str(shown)
