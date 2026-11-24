"""Tomita 7: 0*1*0*1*."""
ALPHABET = ['0', '1']
import re
def answer(seq):
    return "yes" if re.fullmatch("0*1*0*1*", "".join(seq)) else "no"
