"""Vending machine snack counter.

A vending machine starts with 0 cents of credit. Each token is one event:
"dime" inserts 10 cents and "quarter" inserts 25 cents, but the machine never
holds more than 50 cents of credit, so a coin that would raise the credit above
50 is rejected (spat back out) and the credit stays as it was. "buy" presses the
button for a snack costing 35 cents: if the credit is at least 35, one snack is
dispensed and 35 cents are deducted (the change stays as credit); otherwise
nothing happens. "refund" returns all credit, setting it back to 0. The answer
is the total number of snacks dispensed over the whole sequence, written as a
decimal string (e.g. "0", "3").
"""

ALPHABET = ["dime", "quarter", "buy", "refund"]

_COIN_VALUE = {"dime": 10, "quarter": 25}
_CAP = 50
_PRICE = 35


def answer(seq):
    credit = 0
    sold = 0
    for tok in seq:
        if tok in _COIN_VALUE:
            v = _COIN_VALUE[tok]
            if credit + v <= _CAP:
                credit += v
        elif tok == "buy":
            if credit >= _PRICE:
                credit -= _PRICE
                sold += 1
        elif tok == "refund":
            credit = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sold)
