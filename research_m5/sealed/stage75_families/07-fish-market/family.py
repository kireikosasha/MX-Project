"""A fishmonger's stall at the fish market.

The stall opens empty, on fresh ice (cold), and can hold at most 3 crates.
Read the day's events in order: "boat" - a boat lands a crate of fish, which
goes onto the stall unless the stall already holds 3 crates (then the crate is
turned away); "buyer" - a buyer takes one crate if there is any (otherwise
leaves with nothing); "sun" - the sun beats down: if the stall is cold its ice
melts and the stall becomes warm, but if the stall was already warm every crate
on it spoils and is thrown out (the stall stays warm); "ice" - fresh ice is
shovelled on and the stall is cold again (crates are kept). The answer is the
number of crates left at closing and whether the stall is cold or warm,
written like "2 cold" or "0 warm".
"""

ALPHABET = ["boat", "buyer", "sun", "ice"]

CAPACITY = 3


def answer(seq):
    crates = 0
    cold = True
    for tok in seq:
        if tok == "boat":
            if crates < CAPACITY:
                crates += 1
        elif tok == "buyer":
            if crates > 0:
                crates -= 1
        elif tok == "sun":
            if cold:
                cold = False
            else:
                crates = 0
        elif tok == "ice":
            cold = True
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%d %s" % (crates, "cold" if cold else "warm")
