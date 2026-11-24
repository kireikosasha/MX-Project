"""A random automaton of 6 states over 3 letters (random_dfa.py, seed 8800, number 9)."""
ALPHABET = ['a', 'b', 'c']
DELTA = [[3, 4, 5], [1, 2, 3], [3, 5, 5], [5, 0, 5], [1, 5, 0], [4, 4, 4]]
ACCEPT = [False, True, True, True, False, True]
def answer(seq):
    q = 0
    for t in seq:
        q = DELTA[q][ALPHABET.index(t)]
    return 'yes' if ACCEPT[q] else 'no'
