"""DeepMind: is the number of changes ab or ba even."""
ALPHABET = ['a', 'b']
def answer(seq):
    return "yes" if sum(x != y for x, y in zip(seq, seq[1:])) % 2 == 0 else "no"
