"""A random automaton of 6 states over 3 letters (random_dfa.py, seed 8800, number 3)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[3, 3, 4], [5, 5, 1], [5, 5, 0], [5, 1, 0], [5, 3, 2], [0, 0, 3]]
ACCEPT = [True, False, False, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
