"""A random automaton of 5 states over 2 letters (random_dfa.py, seed 8800, number 8)."""
ALPHABET = ['a', 'b']
DELTA = [[0, 2], [4, 2], [3, 1], [2, 3], [3, 3]]
ACCEPT = [True, False, False, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
