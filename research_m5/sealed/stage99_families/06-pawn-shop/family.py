"""Pawn shop at closing time.

The shop opens with an empty vault (items held as loan collateral) and an empty
sale shelf, and reads the day's events in order: "pawn" puts one new item into
the vault; "redeem" gives one vaulted item back to its owner (if the vault is
empty, nothing happens); "forfeit" takes one item out of the vault and puts it
on the sale shelf (if the vault is empty, nothing happens); "buy" sells one item
off the shelf (if the shelf is empty, the buyer leaves and nothing happens).
The answer says where more items are at closing: "vault" if the vault holds
more items than the shelf, "shelf" if the shelf holds more than the vault, and
"even" if they hold the same number (including both empty).
"""

ALPHABET = ["pawn", "redeem", "forfeit", "buy"]


def answer(seq):
    vault = 0
    shelf = 0
    for tok in seq:
        if tok == "pawn":
            vault += 1
        elif tok == "redeem":
            if vault > 0:
                vault -= 1
        elif tok == "forfeit":
            if vault > 0:
                vault -= 1
                shelf += 1
        elif tok == "buy":
            if shelf > 0:
                shelf -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if vault > shelf:
        return "vault"
    if shelf > vault:
        return "shelf"
    return "even"
