"""A random automaton of 8 states over 3 letters (random_dfa.py, seed 8800, number 11)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[3, 3, 4], [0, 4, 4], [1, 5, 6], [1, 3, 3], [4, 7, 1], [0, 2, 4], [3, 6, 2], [2, 5, 4]]
ACCEPT = [True, False, False, True, True, True, False, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
