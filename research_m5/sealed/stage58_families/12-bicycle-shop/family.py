"""Bicycle shop repair rack.

The shop has a repair rack with room for 4 bikes and a shelf that holds at most
3 spare inner tubes; the day starts with an empty rack and 1 tube on the shelf.
Each token is an event of the day, read in order. "drop": a customer leaves a
bike for repair, which goes on the rack if there is room (if the rack is full,
the customer is turned away). "flat": a customer comes in with a flat tyre; if
there is a tube on the shelf it is fitted on the spot (the shelf loses one tube
and nothing goes on the rack), otherwise the bike goes on the rack like a drop
(or is turned away if the rack is full). "tubes": a delivery arrives and the
shelf is refilled to 3 tubes. "fixed": the mechanic finishes a bike and it
leaves the rack (nothing happens if the rack is empty). The answer is the
number of bikes still on the rack at closing time, as a string "0" to "4".
"""

ALPHABET = ["drop", "flat", "tubes", "fixed"]

RACK_SIZE = 4
SHELF_SIZE = 3


def answer(seq):
    rack = 0
    shelf = 1
    for tok in seq:
        if tok == "drop":
            if rack < RACK_SIZE:
                rack += 1
        elif tok == "flat":
            if shelf > 0:
                shelf -= 1
            elif rack < RACK_SIZE:
                rack += 1
        elif tok == "tubes":
            shelf = SHELF_SIZE
        elif tok == "fixed":
            if rack > 0:
                rack -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(rack)
