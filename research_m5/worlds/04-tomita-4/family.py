"""Tomita 4: no three zeros in a row."""
ALPHABET = ['0', '1']
def answer(seq):
    return "no" if "000" in "".join(seq) else "yes"
