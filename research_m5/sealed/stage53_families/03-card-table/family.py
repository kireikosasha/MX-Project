"""Card table turn order.

Four players sit around a card table in the seats north, east, south and west
(clockwise in that order). Play starts with north to act and with play moving
clockwise. The tokens are read left to right, each describing what the player
whose turn it is does: "card" means they play a card and the turn passes to
the next seat in the current direction; "skip" means they play a skip card, so
the next seat in the current direction loses its turn and the turn passes to
the seat after it (two seats along); "flip" means they play a reverse card, so
the direction of play flips (clockwise <-> counter-clockwise) and the turn
passes to the next seat in the new direction; "wait" means they knock and keep
the turn themselves (nothing changes). The answer is the name of the seat whose
turn it is after the whole sequence has been played.
"""

ALPHABET = ["card", "skip", "flip", "wait"]

_SEATS = ["north", "east", "south", "west"]


def answer(seq):
    seat = 0        # index into _SEATS; north starts
    direction = 1   # +1 clockwise, -1 counter-clockwise
    for tok in seq:
        if tok == "card":
            seat = (seat + direction) % 4
        elif tok == "skip":
            seat = (seat + 2 * direction) % 4
        elif tok == "flip":
            direction = -direction
            seat = (seat + direction) % 4
        elif tok == "wait":
            pass
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _SEATS[seat]
