"""A random automaton of 7 states over 2 letters (random_dfa.py, seed 8800, number 4)."""
ALPHABET = ['a', 'b']
DELTA = [[2, 1], [1, 1], [5, 1], [4, 2], [2, 5], [6, 3], [3, 2]]
ACCEPT = [False, True, True, False, False, True, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
