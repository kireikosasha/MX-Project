"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8800, number 7)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[1, 3, 2], [1, 3, 0], [0, 1, 2], [0, 3, 2]]
ACCEPT = [False, True, True, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
