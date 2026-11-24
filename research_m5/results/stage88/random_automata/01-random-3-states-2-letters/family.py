"""A random automaton of 3 states over 2 letters (random_dfa.py, seed 8800, number 0)."""
ALPHABET = ['a', 'b']
DELTA = [[1, 2], [0, 0], [1, 0]]
ACCEPT = [True, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
