"""A windmill's grain hopper.

The mill starts with an empty hopper and the sail brake released.  Reading
the tokens in order: "sack" means the miller pours one sack of grain into the
hopper, which holds at most three sacks (a sack poured into a full hopper
spills and changes nothing); "gust" means a gust of wind turns the sails and,
if the brake is released and the hopper is not empty, the millstones grind one
sack of grain into flour, taking it out of the hopper (with the brake engaged a
gust does nothing); "brake" means the miller engages the sail brake (it stays
engaged if it already was); "lull" means the wind drops and the miller
releases the brake (it stays released if it already was).  The answer is how
much grain is left in the hopper at the end: "empty" (0 sacks), "low" (1),
"high" (2) or "full" (3).
"""

ALPHABET = ["sack", "gust", "brake", "lull"]

_LEVELS = ("empty", "low", "high", "full")


def answer(seq):
    hopper = 0
    braked = False
    for tok in seq:
        if tok == "sack":
            if hopper < 3:
                hopper += 1
        elif tok == "gust":
            if not braked and hopper > 0:
                hopper -= 1
        elif tok == "brake":
            braked = True
        elif tok == "lull":
            braked = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _LEVELS[hopper]
