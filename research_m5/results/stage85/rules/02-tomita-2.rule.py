ALPHABET = ['0', '1']


def answer(seq):
    shown = 'yes'  # what the display shows
    p0 = 0  # a whole number
    for tok in seq:
        if tok == '0':
            effect = {(0, 2): None, (0, 1): 'no'}.get((0, (p0 > 0) - (p0 < 0) + 1,), 'yes')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 1
            p0 = n0
            shown = new
        elif tok == '1':
            effect = {('no',): None}.get((shown,), 'no')
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 0: d0 += 2
            if seen == 1: d0 += -1
            n0 = s0 if s0 is not None else p0 + d0
            p0 = n0
            shown = new
    return str(shown)
