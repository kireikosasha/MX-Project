"""Tomita 2: (10)* - "1 0" repeated."""
ALPHABET = ['0', '1']
def answer(seq):
    s = "".join(seq)
    return "yes" if len(s) % 2 == 0 and s == "10" * (len(s) // 2) else "no"
