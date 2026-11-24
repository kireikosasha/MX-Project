"""The letter seen two steps ago (or none)."""
ALPHABET = ['a', 'b', 'c']
def answer(seq):
    return seq[-2] if len(seq) >= 2 else "none"
