"""Balanced brackets of three kinds: balanced, open, or broken for good."""
ALPHABET = ['(', ')', '[', ']', '{', '}']
def answer(seq):
    stack = []
    for t in seq:
        if t in "([{":
            stack.append(t)
        elif not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[t]:
            return "broken"
    return "balanced" if not stack else "open"
