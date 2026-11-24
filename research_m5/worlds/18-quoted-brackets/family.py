"""Brackets of one kind, where brackets between double quotes do not count: balanced, open, or broken for good."""
ALPHABET = ['(', ')', '"']
def answer(seq):
    depth, quoted = 0, False
    for t in seq:
        if t == '"':
            quoted = not quoted
        elif not quoted:
            depth += 1 if t == "(" else -1
            if depth < 0:
                return "broken"
    return "balanced" if depth == 0 else "open"
