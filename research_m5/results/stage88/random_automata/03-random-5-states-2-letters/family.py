"""A random automaton of 5 states over 2 letters (random_dfa.py, seed 8800, number 2)."""
ALPHABET = ['a', 'b']
DELTA = [[4, 1], [1, 0], [0, 1], [3, 2], [3, 3]]
ACCEPT = [False, True, False, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
