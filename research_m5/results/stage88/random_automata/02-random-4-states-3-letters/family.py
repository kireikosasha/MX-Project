"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8800, number 1)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[2, 3, 0], [0, 1, 0], [1, 2, 2], [3, 1, 1]]
ACCEPT = [True, False, False, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
