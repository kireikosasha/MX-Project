"""A random automaton of 8 states over 3 letters (random_dfa.py, seed 8900, number 17)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[7, 3, 3], [4, 5, 1], [4, 4, 4], [6, 5, 5], [5, 7, 7], [6, 2, 1], [6, 7, 6], [0, 2, 7]]
ACCEPT = [True, False, True, True, False, True, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
