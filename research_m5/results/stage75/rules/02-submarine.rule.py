ALPHABET = ['dive', 'rise', 'blow', 'hatch']


def answer(seq):
    shown = 'surface'  # what the display shows
    for tok in seq:
        if tok == 'dive':
            effect = {('surface',): 'periscope', ('periscope',): 'cruise', ('cruise',): 'deep'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'rise':
            effect = {('periscope',): 'surface', ('cruise',): 'periscope', ('deep',): 'cruise'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'blow':
            effect = {('open',): None, ('surface',): None}.get((shown,), 'surface')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
        elif tok == 'hatch':
            effect = {('surface',): 'open', ('open',): 'surface'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
