ALPHABET = ['0', '1']


def answer(seq):
    shown = 0  # what the display shows
    for tok in seq:
        if tok == '0':
            effect = {(0,): 0, (2,): -1}.get((min(max(shown, 0), 2),), 1)
            new = shown + effect
            seen = effect
            shown = new
        elif tok == '1':
            effect = {(2,): 0, (1,): -1}.get((min(max(shown, 0), 2),), 1)
            new = shown + effect
            seen = effect
            shown = new
    return str(shown)
