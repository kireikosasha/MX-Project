"""A random automaton of 6 states over 3 letters (random_dfa.py, seed 8900, number 15)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[4, 5, 1], [4, 2, 2], [5, 0, 0], [1, 1, 2], [1, 1, 1], [3, 5, 2]]
ACCEPT = [False, True, True, False, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
