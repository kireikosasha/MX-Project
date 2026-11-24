"""A random automaton of 5 states over 2 letters (random_dfa.py, seed 8800, number 14)."""
ALPHABET = ['a', 'b']
DELTA = [[0, 1], [3, 2], [3, 4], [4, 4], [3, 1]]
ACCEPT = [True, True, False, False, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
