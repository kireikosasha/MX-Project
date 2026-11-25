"""A random automaton of 3 states over 2 letters (random_dfa.py, seed 8900, number 18)."""
ALPHABET = ['a', 'b']
DELTA = [[0, 2], [0, 1], [1, 1]]
ACCEPT = [True, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
