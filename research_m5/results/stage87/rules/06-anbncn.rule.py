ALPHABET = ['a', 'b', 'c']


def answer(seq):
    shown = 'maybe'  # what the display shows
    p0 = 0  # a counter, not below zero
    p1 = 0  # a counter, not below zero
    p2 = 0  # a level from 0 to 5
    for tok in seq:
        if tok == 'a':
            effect = {('yes', 1): 'no', ('maybe', 1): 'no'}.get((shown, int(p0 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n1 = p1 + 1
            n2 = p2 + 2 if 0 <= p2 + 2 <= 5 else p2
            p1 = n1
            p2 = n2
            shown = new
        elif tok == 'b':
            effect = {('maybe', 0): 'no', ('yes', 0): 'no'}.get((shown, int(p1 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 1
            n1 = max(0, p1 - 1)
            n2 = p2 + 3 if 0 <= p2 + 3 <= 5 else p2
            p0 = n0
            p1 = n1
            p2 = n2
            shown = new
        elif tok == 'c':
            effect = {('no', 1): None, ('no', 0): None, ('maybe', 1): 'yes'}.get((shown, int(p2 == 5),), 'no')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
