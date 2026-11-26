"""Auction house: two bidders, Ann and Bob, compete for a series of lots.

Each lot opens at price 0 with no high bidder. The token "ann" (or "bob") is
a bid by that person: if they are not already the high bidder, the price of
the current lot goes up by 1 and they become the high bidder; if they already
hold the high bid, the token changes nothing (you cannot outbid yourself).
The token "gavel" closes the current lot: if someone holds the high bid, the
lot is sold to them and the current price is added to that bidder's total
spending; if nobody has bid, the lot goes unsold. Either way a fresh lot then
opens at price 0 with no high bidder. A lot still open when the sequence ends
is never sold and counts for nobody. The answer is who has spent more in total
over all sold lots: "ann", "bob", or "even" if their totals are equal.
"""

ALPHABET = ["ann", "bob", "gavel"]


def answer(seq):
    price = 0
    high = None
    spent = {"ann": 0, "bob": 0}
    for tok in seq:
        if tok == "gavel":
            if high is not None:
                spent[high] += price
            price = 0
            high = None
        elif tok in spent:
            if high != tok:
                price += 1
                high = tok
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if spent["ann"] > spent["bob"]:
        return "ann"
    if spent["bob"] > spent["ann"]:
        return "bob"
    return "even"
