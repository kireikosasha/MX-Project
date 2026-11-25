"""A random automaton of 8 states over 3 letters (random_dfa.py, seed 8900, number 11)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[1, 3, 2], [7, 3, 0], [0, 0, 6], [3, 6, 0], [7, 2, 7], [6, 5, 4], [3, 5, 2], [7, 3, 3]]
ACCEPT = [True, False, False, True, False, True, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
