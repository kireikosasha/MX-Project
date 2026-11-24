"""Which letter has been seen more often, a or b (or neither)."""
ALPHABET = ['a', 'b', 'c']
def answer(seq):
    a, b = seq.count("a"), seq.count("b")
    return "a" if a > b else "b" if b > a else "tie"
