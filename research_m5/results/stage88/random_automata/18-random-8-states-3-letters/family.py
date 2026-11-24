"""A random automaton of 8 states over 3 letters (random_dfa.py, seed 8800, number 17)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[7, 5, 5], [4, 2, 0], [4, 2, 1], [4, 7, 0], [0, 6, 5], [2, 3, 2], [3, 0, 7], [3, 1, 6]]
ACCEPT = [True, False, False, True, True, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
