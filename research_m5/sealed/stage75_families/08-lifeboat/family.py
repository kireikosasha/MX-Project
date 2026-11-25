"""A lifeboat in a rough sea: how much water is in the bottom of the boat.

The crew of the lifeboat is always doing one of two jobs, rowing or bailing;
they start out rowing, and the boat starts out dry (water level 0, on a scale
from 0 to 3). Each token is something that happens, read in order. "swap": the
crew drop what they are doing and switch to the other job (rowing becomes
bailing, bailing becomes rowing). "wave": a wave breaks over the side; if the
crew are rowing, the water rises by one level (it never goes above 3), but if
they are bailing they keep pace with it and the level stays the same. "lull":
a quiet moment; if the crew are bailing, they lower the water by one level (it
never goes below 0), but if they are rowing nothing changes. The answer is the
water level after the last token, named "dry" (0), "damp" (1), "wet" (2) or
"awash" (3).
"""

ALPHABET = ["wave", "lull", "swap"]

_LEVELS = ["dry", "damp", "wet", "awash"]


def answer(seq):
    bailing = False
    water = 0
    for tok in seq:
        if tok == "swap":
            bailing = not bailing
        elif tok == "wave":
            if not bailing and water < 3:
                water += 1
        elif tok == "lull":
            if bailing and water > 0:
                water -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _LEVELS[water]
