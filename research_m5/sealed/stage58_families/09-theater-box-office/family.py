"""Theater box office for a tiny studio theater with 5 seats.

The house starts empty (0 seats sold).  Read the tokens in order: "sell" sells
one ticket if at least one seat is still free (otherwise the buyer is turned
away and nothing changes); "pair" sells two tickets to a couple, but only if at
least two seats are free (a couple never splits up, so with 0 or 1 free seats
nothing changes); "refund" returns one ticket, freeing one seat, if at least
one seat is sold (with an empty house nothing changes).  The answer is the
number of seats sold when the sequence ends, written as a digit "0" to "5".
"""

ALPHABET = ["sell", "pair", "refund"]

CAPACITY = 5


def answer(seq):
    sold = 0
    for tok in seq:
        if tok == "sell":
            if sold < CAPACITY:
                sold += 1
        elif tok == "pair":
            if sold <= CAPACITY - 2:
                sold += 2
        elif tok == "refund":
            if sold > 0:
                sold -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sold)
