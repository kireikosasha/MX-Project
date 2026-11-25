"""A random automaton of 8 states over 3 letters (random_dfa.py, seed 8900, number 5)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[1, 4, 2], [5, 6, 0], [6, 2, 4], [0, 6, 6], [7, 6, 1], [0, 0, 4], [4, 1, 5], [7, 3, 5]]
ACCEPT = [True, True, False, True, False, True, False, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
