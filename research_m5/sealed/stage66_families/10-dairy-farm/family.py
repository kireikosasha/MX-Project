"""Dairy farm creamery rating.

A farm has a bulk milk tank (empty at the start, holds at most 3 units) with
a chiller that starts out working, and the creamery rates the farm on four
grades, poor < fair < good < prime, starting at "fair". "milk" pours one
unit into the tank (milking into a full tank spills, so it stays at 3).
"calf" feeds one unit from the tank to a calf if the tank is not empty.
"storm" knocks the chiller out and "repair" gets it working again (each is
harmless if the chiller is already in that state). "truck" is the
collection truck: if the tank holds any milk, the whole load is taken and
the tank is left empty; if the chiller is working at that moment the load
is accepted and the rating goes up one grade (it stays at "prime" if
already there), and if the chiller is out the load is rejected and the
rating goes down one grade (it stays at "poor" if already there). A truck
that finds the tank empty changes nothing. The answer is the farm's rating
after the whole sequence.
"""

ALPHABET = ["milk", "calf", "storm", "repair", "truck"]

CAPACITY = 3
GRADES = ["poor", "fair", "good", "prime"]


def answer(seq):
    level = 0
    chiller_ok = True
    rating = 1
    for tok in seq:
        if tok == "milk":
            if level < CAPACITY:
                level += 1
        elif tok == "calf":
            if level > 0:
                level -= 1
        elif tok == "storm":
            chiller_ok = False
        elif tok == "repair":
            chiller_ok = True
        elif tok == "truck":
            if level > 0:
                if chiller_ok:
                    rating = min(rating + 1, len(GRADES) - 1)
                else:
                    rating = max(rating - 1, 0)
                level = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return GRADES[rating]
