"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8900, number 13)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[2, 3, 2], [2, 1, 3], [1, 3, 2], [3, 1, 3]]
ACCEPT = [True, False, True, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
