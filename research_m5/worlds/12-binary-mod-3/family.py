"""A binary number read from its highest digit: its remainder modulo 3."""
ALPHABET = ['0', '1']
def answer(seq):
    r = 0
    for t in seq:
        r = (2 * r + int(t)) % 3
    return str(r)
