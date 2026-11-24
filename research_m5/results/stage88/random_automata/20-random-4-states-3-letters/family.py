"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8800, number 19)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[0, 0, 1], [0, 2, 2], [3, 2, 0], [0, 2, 2]]
ACCEPT = [False, True, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
