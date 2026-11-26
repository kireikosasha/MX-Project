ALPHABET = ['train', 'clear', 'car']


def answer(seq):
    shown = 'open'  # what the display shows
    p0 = 0  # a counter, not below zero
    for tok in seq:
        if tok == 'train':
            effect = {(0, 1): None}.get((0, int(p0 >= 1),), 'closed:0')
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + 1
            p0 = n0
            shown = new
        elif tok == 'clear':
            effect = {('closed:0', 0): 'open', ('closed:1', 0): 'open', ('closed:4', 0): 'open', ('closed:3', 0): 'open', ('closed:2', 0): 'open', ('closed:5', 0): 'open', ('closed:6', 0): 'open', ('closed:7', 0): 'open'}.get((shown, int(p0 >= 2),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = max(0, p0 - 1)
            p0 = n0
            shown = new
        elif tok == 'car':
            effect = {('open',): None, ('closed:0',): 'closed:1', ('closed:1',): 'closed:2', ('closed:2',): 'closed:3', ('closed:3',): 'closed:4', ('closed:4',): 'closed:5', ('closed:5',): 'closed:6', ('closed:6',): 'closed:7', ('closed:7',): 'closed:8', ('closed:9',): 'closed:10', ('closed:10',): 'closed:11', ('closed:11',): 'closed:12', ('closed:12',): 'closed:13', ('closed:13',): 'closed:14', ('closed:14',): 'closed:15'}.get((shown,), 'closed:9')
            new = shown if effect is None else effect
            seen = int(new != shown)
            shown = new
    return str(shown)
