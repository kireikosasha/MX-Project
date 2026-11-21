"""Observatory night log.

Each token is an event during one night at a small observatory. The dome starts
shut and the sky starts clear. "open" opens the dome and "shut" closes it (doing
either when the dome is already in that state changes nothing); "cloud" makes the
sky cloudy and "clear" makes it clear again (likewise idempotent). "snap" takes
one exposure: it counts as a good frame only if, at that moment, the dome is open
and the sky is clear; otherwise it is wasted. The answer is the number of good
frames taken over the whole night, written as a decimal string (e.g. "0", "3").
"""

ALPHABET = ["open", "shut", "cloud", "clear", "snap"]


def answer(seq):
    dome_open = False
    sky_clear = True
    good = 0
    for tok in seq:
        if tok == "open":
            dome_open = True
        elif tok == "shut":
            dome_open = False
        elif tok == "cloud":
            sky_clear = False
        elif tok == "clear":
            sky_clear = True
        elif tok == "snap":
            if dome_open and sky_clear:
                good += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(good)
