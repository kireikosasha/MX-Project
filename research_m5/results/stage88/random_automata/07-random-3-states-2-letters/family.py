"""A random automaton of 3 states over 2 letters (random_dfa.py, seed 8800, number 6)."""
ALPHABET = ['a', 'b']
DELTA = [[2, 0], [1, 1], [1, 2]]
ACCEPT = [True, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
