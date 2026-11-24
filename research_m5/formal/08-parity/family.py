"""Parity (Bhattamishra, Ahuja, Goyal 2020): strings of 0 and 1; "yes" when the number of 1s is even, else "no"."""
ALPHABET = ['0', '1']
def answer(seq):
    return "yes" if seq.count('1') % 2 == 0 else "no"
