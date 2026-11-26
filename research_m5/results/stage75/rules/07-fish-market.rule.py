ALPHABET = ['boat', 'buyer', 'sun', 'ice']


def answer(seq):
    shown = '0 cold'  # what the display shows
    for tok in seq:
        if tok == 'boat':
            effect = {('0 cold',): '1 cold', ('0 warm',): '1 warm', ('1 cold',): '2 cold', ('2 cold',): '3 cold', ('1 warm',): '2 warm', ('2 warm',): '3 warm'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'buyer':
            effect = {('1 warm',): '0 warm', ('1 cold',): '0 cold', ('2 warm',): '1 warm', ('2 cold',): '1 cold', ('3 cold',): '2 cold', ('3 warm',): '2 warm'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'sun':
            effect = {('1 cold',): '1 warm', ('0 warm',): None, ('2 cold',): '2 warm', ('3 cold',): '3 warm'}.get((shown,), '0 warm')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'ice':
            effect = {('1 warm',): '1 cold', ('0 warm',): '0 cold', ('2 warm',): '2 cold', ('3 warm',): '3 cold'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
