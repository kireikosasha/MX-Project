"""A random automaton of 8 states over 3 letters (random_dfa.py, seed 8800, number 5)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[4, 4, 7], [5, 5, 6], [1, 0, 0], [1, 4, 0], [2, 7, 3], [6, 0, 4], [1, 5, 4], [0, 4, 0]]
ACCEPT = [False, False, False, True, False, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
