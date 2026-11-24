"""DeepMind (Deletang et al. 2023): a walker on a circle of five; 0 steps back, 1 stays, 2 steps forward; where is it."""
ALPHABET = ['0', '1', '2']
def answer(seq):
    return str(sum(int(t) - 1 for t in seq) % 5)
