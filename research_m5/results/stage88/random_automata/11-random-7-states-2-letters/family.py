"""A random automaton of 7 states over 2 letters (random_dfa.py, seed 8800, number 10)."""
ALPHABET = ['a', 'b']
DELTA = [[5, 4], [5, 3], [0, 0], [0, 5], [1, 6], [2, 4], [6, 6]]
ACCEPT = [False, True, True, True, False, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
