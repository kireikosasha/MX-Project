"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8900, number 1)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[1, 2, 3], [3, 1, 2], [2, 0, 0], [3, 3, 1]]
ACCEPT = [False, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
