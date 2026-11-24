ALPHABET = ['a', 'b', 'c']


def answer(seq):
    shown = 'tie'  # what the display shows
    p0 = 0  # a whole number
    for tok in seq:
        if tok == 'a':
            effect = {('b', 0): 'tie', ('tie', 0): 'a'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 0: d0 += -1
            n0 = s0 if s0 is not None else p0 + d0
            p0 = n0
            shown = new
        elif tok == 'b':
            effect = {('tie', 1): 'b', ('a', 1): 'tie'}.get((shown, (p0 > 0) - (p0 < 0) + 1,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 0: d0 += 1
            n0 = s0 if s0 is not None else p0 + d0
            p0 = n0
            shown = new
        elif tok == 'c':
            effect = None
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
