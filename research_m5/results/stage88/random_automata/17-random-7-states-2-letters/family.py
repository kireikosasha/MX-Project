"""A random automaton of 7 states over 2 letters (random_dfa.py, seed 8800, number 16)."""
ALPHABET = ['a', 'b']
DELTA = [[4, 5], [6, 0], [0, 5], [2, 5], [1, 0], [1, 4], [2, 3]]
ACCEPT = [True, True, True, True, True, True, False]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
