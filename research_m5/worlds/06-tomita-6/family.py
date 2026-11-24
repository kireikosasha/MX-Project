"""Tomita 6: the number of zeros minus the number of ones is a multiple of three."""
ALPHABET = ['0', '1']
def answer(seq):
    return "yes" if (seq.count("0") - seq.count("1")) % 3 == 0 else "no"
