"""A random automaton of 5 states over 2 letters (random_dfa.py, seed 8900, number 14)."""
ALPHABET = ['a', 'b']
DELTA = [[3, 4], [0, 4], [1, 1], [3, 2], [4, 2]]
ACCEPT = [True, False, False, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
