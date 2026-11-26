ALPHABET = ['cab', 'fare', 'bus']


def answer(seq):
    shown = 'empty'  # what the display shows
    for tok in seq:
        if tok == 'cab':
            effect = {('empty',): '1 cabs', ('1 cabs',): '2 cabs', ('2 cabs',): '3 cabs', ('2 waiting',): '1 waiting', ('3 waiting',): '2 waiting', ('3 cabs',): '4 cabs', ('4 cabs',): None, ('5 waiting',): '4 waiting', ('4 waiting',): '3 waiting'}.get((shown,), 'empty')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'fare':
            effect = {('empty',): '1 waiting', ('1 waiting',): '2 waiting', ('2 waiting',): '3 waiting', ('3 waiting',): '4 waiting', ('2 cabs',): '1 cabs', ('4 cabs',): '3 cabs', ('3 cabs',): '2 cabs', ('4 waiting',): '5 waiting', ('5 waiting',): None}.get((shown,), 'empty')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'bus':
            effect = {('empty',): None, ('2 cabs',): None, ('1 cabs',): None, ('3 cabs',): None, ('4 cabs',): None}.get((shown,), 'empty')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
