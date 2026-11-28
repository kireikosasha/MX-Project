"""Kite festival: count the kites still flying at the end of the day.

The wind has four levels, 0 (dead calm), 1 (light), 2 (fresh) and 3 (storm),
and the festival opens at level 1 with no kites in the air.  Read the tokens
in order.  "launch": if the wind is 1 or 2 a new kite goes up (one more kite
aloft); at 0 it will not lift and at 3 it is torn away at once, so nothing
changes.  "gust": if the wind is below 3 it rises by one level; if it is
already 3 it stays at 3 and one kite in the air snaps loose (one fewer aloft,
if any are up).  "lull": if the wind is above 0 it drops by one level, and if
that brings it down to 0 one kite in the air falls (one fewer aloft, if any
are up); if the wind is already 0 nothing happens.  The answer is the number
of kites still in the air after the last token, written as a decimal number
(for example "0" or "7").
"""

ALPHABET = ["launch", "gust", "lull"]


def answer(seq):
    wind = 1
    aloft = 0
    for tok in seq:
        if tok == "launch":
            if wind in (1, 2):
                aloft += 1
        elif tok == "gust":
            if wind < 3:
                wind += 1
            elif aloft > 0:
                aloft -= 1
        elif tok == "lull":
            if wind > 0:
                wind -= 1
                if wind == 0 and aloft > 0:
                    aloft -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(aloft)
