"""Ferry ledger.

A small ferry starts docked at the west bank, empty, with 4 units of deck
space.  Reading the tokens in order: "walker" means a foot passenger tries to
board (needs 1 unit) and "car" means a car tries to board (needs 2 units); if
there is not enough free space left on deck, that passenger or car is turned
away and nothing changes.  "sail" means the ferry crosses to the other bank,
everything aboard is unloaded there and its units are credited as delivered to
that arrival bank, and the ferry is then empty (an empty ferry still crosses).
"storm" means the trip is called off: everything aboard gets off back onto the
bank where the ferry is docked, nothing is delivered, the ferry stays put and
is empty.  The answer compares the total units ever delivered to each bank:
"east" if the east bank received more, "west" if the west bank received more,
and "tie" if both received the same amount (including nothing at all).
"""

ALPHABET = ["walker", "car", "sail", "storm"]

_CAPACITY = 4
_SIZE = {"walker": 1, "car": 2}


def answer(seq):
    at_east = False  # ferry starts at the west bank
    load = 0
    balance = 0  # delivered to east minus delivered to west
    for tok in seq:
        if tok in _SIZE:
            need = _SIZE[tok]
            if load + need <= _CAPACITY:
                load += need
        elif tok == "sail":
            at_east = not at_east
            if at_east:
                balance += load
            else:
                balance -= load
            load = 0
        elif tok == "storm":
            load = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "east"
    if balance < 0:
        return "west"
    return "tie"
