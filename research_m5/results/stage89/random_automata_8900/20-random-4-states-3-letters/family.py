"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8900, number 19)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[0, 3, 0], [3, 2, 3], [0, 2, 2], [0, 1, 1]]
ACCEPT = [False, False, True, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
