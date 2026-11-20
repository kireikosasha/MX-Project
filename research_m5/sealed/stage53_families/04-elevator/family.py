"""Elevator with a door interlock.

An elevator serves a building with floors 0 through 5 and starts at floor 0
with its doors closed. Read the tokens in order: "open" opens the doors and
"close" closes them (each has no effect if the doors are already in that
state); "up" moves the car one floor higher and "down" one floor lower, but
only while the doors are closed -- a move requested while the doors are open
is ignored for safety, and a move past the top floor (5) or below the ground
floor (0) is also ignored. The answer is the car's final floor followed by
the final door state, written as e.g. "3 open" or "0 closed".
"""

ALPHABET = ["up", "down", "open", "close"]

TOP = 5


def answer(seq):
    floor = 0
    is_open = False
    for tok in seq:
        if tok == "open":
            is_open = True
        elif tok == "close":
            is_open = False
        elif tok == "up":
            if not is_open and floor < TOP:
                floor += 1
        elif tok == "down":
            if not is_open and floor > 0:
                floor -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%d %s" % (floor, "open" if is_open else "closed")
