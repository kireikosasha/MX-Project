"""A random automaton of 6 states over 3 letters (random_dfa.py, seed 8900, number 9)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[0, 5, 4], [5, 0, 4], [2, 3, 4], [3, 2, 1], [5, 2, 4], [0, 2, 2]]
ACCEPT = [True, False, True, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
