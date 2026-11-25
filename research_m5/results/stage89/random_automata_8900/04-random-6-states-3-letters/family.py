"""A random automaton of 6 states over 3 letters (random_dfa.py, seed 8900, number 3)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[3, 5, 5], [3, 4, 1], [3, 3, 2], [1, 0, 2], [3, 1, 3], [1, 0, 0]]
ACCEPT = [False, False, True, True, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
