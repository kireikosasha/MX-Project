"""Fishing boat hold.

A small fishing boat starts at sea with its net on deck and an empty hold that
can take at most 3 crates. Reading the tokens in order: "cast" puts the net in
the water (nothing happens if it is already in the water); "haul" pulls the net
up and, only if it was in the water, adds one crate of fish to the hold (a full
hold stays at 3 and the extra catch is thrown back); "gull" means a gull steals
one crate from the hold if there is any; "port" means the boat lands and sells
everything, so the hold becomes empty (the net stays where it is). The answer
is the number of crates in the hold after the last token, as a string "0".."3".
"""

ALPHABET = ["cast", "haul", "gull", "port"]

CAPACITY = 3


def answer(seq):
    net_in_water = False
    crates = 0
    for tok in seq:
        if tok == "cast":
            net_in_water = True
        elif tok == "haul":
            if net_in_water:
                crates = min(CAPACITY, crates + 1)
            net_in_water = False
        elif tok == "gull":
            if crates > 0:
                crates -= 1
        elif tok == "port":
            crates = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(crates)
