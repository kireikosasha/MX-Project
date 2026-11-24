"""A random automaton of 4 states over 3 letters (random_dfa.py, seed 8800, number 13)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[3, 2, 1], [0, 3, 0], [0, 2, 0], [2, 1, 1]]
ACCEPT = [True, True, False, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
