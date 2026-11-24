ALPHABET = ['(', ')', '[', ']', '{', '}']


def answer(seq):
    shown = 'balanced'  # what the display shows
    p0 = []  # a stack, the last put on top
    for tok in seq:
        if tok == '(':
            effect = {('balanced',): 'open'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + ['(']
            p0 = n0
            shown = new
        elif tok == ')':
            effect = {('open', 10): 'broken', ('balanced', 0): 'broken', ('open', 7): 'broken', ('open', 11): 'broken', ('open', 8): 'broken', ('open', 4): 'balanced'}.get((shown, (({'(': 1, '[': 2, '{': 3}[p0[-1]] if p0 else 0) * 3 + min(len(p0), 2)),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0[:-1]
            p0 = n0
            shown = new
        elif tok == '[':
            effect = {('balanced',): 'open'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + ['[']
            p0 = n0
            shown = new
        elif tok == ']':
            effect = {('balanced', 0): 'broken', ('open', 10): 'broken', ('open', 7): 'balanced', ('open', 11): 'broken', ('open', 4): 'broken', ('open', 5): 'broken'}.get((shown, (({'(': 1, '[': 2, '{': 3}[p0[-1]] if p0 else 0) * 3 + min(len(p0), 2)),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0[:-1]
            p0 = n0
            shown = new
        elif tok == '{':
            effect = {('balanced',): 'open'}.get((shown,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0 + ['{']
            p0 = n0
            shown = new
        elif tok == '}':
            effect = {('open', 10): 'balanced', ('balanced', 0): 'broken', ('open', 4): 'broken', ('open', 5): 'broken', ('open', 8): 'broken', ('open', 7): 'broken'}.get((shown, (({'(': 1, '[': 2, '{': 3}[p0[-1]] if p0 else 0) * 3 + min(len(p0), 2)),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = p0[:-1]
            p0 = n0
            shown = new
    return str(shown)
