"""A random automaton of 7 states over 2 letters (random_dfa.py, seed 8900, number 16)."""
ALPHABET = ['a', 'b']
DELTA = [[1, 4], [6, 4], [2, 3], [4, 1], [0, 5], [0, 6], [4, 2]]
ACCEPT = [False, True, False, True, False, False, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
