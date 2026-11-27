"""Cheese cellar: which cheese sold more ripe wheels.

The cellar starts empty, with one long shelf where wheels wait in the order
they were made, and a calendar that starts at month 0. "brie" and "gouda" each
put a new wheel of that cheese at the back of the shelf, made in the current
month. "month" makes one month pass, so every wheel on the shelf gets one month
older. "sell" is a customer asking for the wheel at the front of the shelf (the
oldest one still there): if that wheel has aged at least 2 months it is sold
and leaves the shelf, counting as one sale for its cheese; if it is younger
than 2 months, the customer refuses it, leaves with nothing, and the wheel
stays at the front; if the shelf is empty nothing happens. After the whole
sequence, the answer is "brie" if more brie wheels were sold than gouda wheels,
"gouda" if more gouda wheels were sold than brie wheels, and "tie" if the two
counts are equal (including when nothing was sold).
"""

ALPHABET = ["brie", "gouda", "month", "sell"]

RIPE_AGE = 2


def answer(seq):
    clock = 0          # months passed so far
    shelf = []         # (cheese, month made) for each wheel, in shelf order
    front = 0          # index in `shelf` of the wheel at the front
    sold = {"brie": 0, "gouda": 0}
    for tok in seq:
        if tok == "brie" or tok == "gouda":
            shelf.append((tok, clock))
        elif tok == "month":
            clock += 1
        elif tok == "sell":
            if front < len(shelf):
                cheese, made = shelf[front]
                if clock - made >= RIPE_AGE:
                    sold[cheese] += 1
                    front += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if sold["brie"] > sold["gouda"]:
        return "brie"
    if sold["gouda"] > sold["brie"]:
        return "gouda"
    return "tie"
