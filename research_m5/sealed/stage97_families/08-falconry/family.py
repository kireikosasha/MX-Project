"""A day at the falconry mews.

Each token is one event of the day, read in order, and the falconer keeps two
counts that both start at zero: the number of birds aloft and the number of
quarry in the game bag. "cast" puts one more bird in the air. "lure" calls one
bird back down to the mews, if any bird is aloft (otherwise it does nothing).
"kill" adds one quarry to the bag, but only while at least one bird is aloft
(with no bird up, nothing is caught). "feed" takes one quarry out of the bag
to feed the birds in the mews, if the bag is not empty (otherwise it does
nothing). At dusk the answer is "home" if no bird is aloft; otherwise each
quarry left in the bag can be garnished on the lure to bring one bird down, so
the answer is "lured" if the bag holds at least as many quarry as there are
birds aloft, and "lost" if it holds fewer.
"""

ALPHABET = ["cast", "lure", "kill", "feed"]


def answer(seq):
    aloft = 0
    bag = 0
    for tok in seq:
        if tok == "cast":
            aloft += 1
        elif tok == "lure":
            if aloft > 0:
                aloft -= 1
        elif tok == "kill":
            if aloft > 0:
                bag += 1
        elif tok == "feed":
            if bag > 0:
                bag -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if aloft == 0:
        return "home"
    if bag >= aloft:
        return "lured"
    return "lost"
