"""A random automaton of 5 states over 2 letters (random_dfa.py, seed 8900, number 8)."""
ALPHABET = ['a', 'b']
DELTA = [[3, 1], [1, 4], [0, 4], [2, 0], [0, 3]]
ACCEPT = [True, False, True, True, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
