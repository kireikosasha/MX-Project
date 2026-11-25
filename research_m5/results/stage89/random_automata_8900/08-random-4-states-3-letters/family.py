"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8900, number 7)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[0, 0, 3], [1, 2, 3], [2, 2, 2], [1, 0, 0]]
ACCEPT = [False, False, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
