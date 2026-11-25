"""A random automaton of 5 states over 2 letters (random_dfa.py, seed 8900, number 2)."""
ALPHABET = ['a', 'b']
DELTA = [[1, 4], [2, 1], [0, 3], [0, 3], [3, 2]]
ACCEPT = [True, False, True, False, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
