"""Tomita (1982), language 1: only ones."""
ALPHABET = ['0', '1']
def answer(seq):
    return "yes" if all(t == "1" for t in seq) else "no"
