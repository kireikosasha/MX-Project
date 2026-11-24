ALPHABET = ['car', 'van', 'leave', 'tow']


def answer(seq):
    shown = 4  # what the display shows
    p0 = []  # a line of at most 5, first come first served
    p1 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'car':
            effect = {(0,): 0}.get((min(max(shown, 0), 2),), -1)
            new = shown + effect
            seen = effect
            if seen == -1: n0 = p0 + ['car'] if len(p0) < 5 else p0
            p0 = n0
            shown = new
        elif tok == 'van':
            effect = {(2,): -2}.get((min(max(shown, 0), 2),), 0)
            new = shown + effect
            seen = effect
            if seen == -2: n0 = p0 + ['van'] if len(p0) < 5 else p0
            d1, s1 = 0, None
            if seen == -2: d1 += 2
            n1 = s1 if s1 is not None else max(0, min(10 ** 9, p1 + d1))
            p0 = n0
            p1 = n1
            shown = new
        elif tok == 'leave':
            effect = {(0, 1): 1, (0, 0): 0}.get((0, ({'car': 1, 'van': 2}[p0[0]] if p0 else 0),), 2)
            new = shown + effect
            seen = effect
            n0 = p0[1:]
            d1, s1 = 0, None
            if seen == 2: d1 += -2
            n1 = s1 if s1 is not None else max(0, min(10 ** 9, p1 + d1))
            p0 = n0
            p1 = n1
            shown = new
        elif tok == 'tow':
            effect = p1
            new = shown + effect
            seen = effect
            n0 = [x for x in p0 if x != 'van']
            n1 = 0
            p0 = n0
            p1 = n1
            shown = new
    return str(shown)
