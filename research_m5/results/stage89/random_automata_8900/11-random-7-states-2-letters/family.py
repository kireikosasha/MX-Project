"""A random automaton of 7 states over 2 letters (random_dfa.py, seed 8900, number 10)."""
ALPHABET = ['a', 'b']
DELTA = [[0, 5], [3, 4], [0, 1], [5, 6], [4, 4], [3, 2], [3, 1]]
ACCEPT = [True, True, True, True, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
