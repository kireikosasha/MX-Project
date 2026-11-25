"""A random automaton of 7 states over 2 letters (random_dfa.py, seed 8900, number 4)."""
ALPHABET = ['a', 'b']
DELTA = [[3, 3], [3, 0], [4, 4], [6, 5], [6, 1], [6, 6], [2, 2]]
ACCEPT = [True, False, True, False, True, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
