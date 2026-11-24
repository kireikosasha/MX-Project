"""A random automaton of 6 states over 3 letters (random_dfa.py, seed 8800, number 15)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[3, 5, 0], [3, 4, 3], [5, 3, 5], [0, 3, 4], [0, 3, 1], [5, 2, 1]]
ACCEPT = [False, True, False, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
