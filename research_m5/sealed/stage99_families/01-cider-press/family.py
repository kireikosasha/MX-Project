"""Cider press: count the bottles filled.

Read the tokens left to right while keeping two running amounts, the number of
apples waiting in the hopper and the number of cups of juice in the vat, both
starting at zero, plus a tally of bottles filled. "apple" drops one apple into
the hopper. "wasp" spoils one apple in the hopper, removing it (nothing happens
if the hopper is empty). "crank" turns the press: if the hopper holds at least
one apple, one apple is crushed into one cup of juice added to the vat; with an
empty hopper the crank does nothing. "bottle" fills a bottle: if the vat holds
at least two cups, two cups are poured out and the bottle tally goes up by
one; otherwise nothing happens. The answer is the final number of bottles
filled, written as a decimal string (for example "0", "3").
"""

ALPHABET = ["apple", "wasp", "crank", "bottle"]


def answer(seq):
    hopper = 0
    vat = 0
    bottles = 0
    for tok in seq:
        if tok == "apple":
            hopper += 1
        elif tok == "wasp":
            if hopper > 0:
                hopper -= 1
        elif tok == "crank":
            if hopper >= 1:
                hopper -= 1
                vat += 1
        elif tok == "bottle":
            if vat >= 2:
                vat -= 2
                bottles += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(bottles)
