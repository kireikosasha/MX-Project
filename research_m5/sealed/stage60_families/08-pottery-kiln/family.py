"""Pottery kiln firing count.

The kiln starts cold, at heat level 0, and its heat can only be 0, 1, 2 or 3.
Reading the sequence left to right: "stoke" raises the heat by one (it cannot go
above 3), "vent" lowers it by one (it cannot go below 0), and "pot" places one
pot in the kiln, which is fired at whatever the heat is at that moment and does
not change the heat. A pot fired at heat exactly 2 comes out good; at heat 0 or 1
it stays underfired and at heat 3 it cracks. The answer is the number of good
pots in the whole sequence, written as a decimal string ("0" if there are none).
"""

ALPHABET = ["stoke", "vent", "pot"]


def answer(seq):
    heat = 0
    good = 0
    for tok in seq:
        if tok == "stoke":
            if heat < 3:
                heat += 1
        elif tok == "vent":
            if heat > 0:
                heat -= 1
        elif tok == "pot":
            if heat == 2:
                good += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(good)
