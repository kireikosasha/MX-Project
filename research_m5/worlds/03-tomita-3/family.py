"""Tomita 3: no odd run of ones followed at once by an odd run of zeros (the runs as they stand)."""
ALPHABET = ['0', '1']
import itertools
def answer(seq):
    runs = [(k, len(list(g))) for k, g in itertools.groupby(seq)]
    for (k1, n1), (k2, n2) in zip(runs, runs[1:]):
        if k1 == "1" and n1 % 2 == 1 and k2 == "0" and n2 % 2 == 1:
            return "no"
    return "yes"
