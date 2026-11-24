"""Balanced brackets of one kind: balanced, open, or broken for good once a bracket closes that was never opened."""
ALPHABET = ['(', ')']
def answer(seq):
    depth = 0
    for t in seq:
        depth += 1 if t == "(" else -1
        if depth < 0:
            return "broken"
    return "balanced" if depth == 0 else "open"
