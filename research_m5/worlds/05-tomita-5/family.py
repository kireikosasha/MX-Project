"""Tomita 5: an even number of zeros and an even number of ones."""
ALPHABET = ['0', '1']
def answer(seq):
    return "yes" if seq.count("0") % 2 == 0 and seq.count("1") % 2 == 0 else "no"
