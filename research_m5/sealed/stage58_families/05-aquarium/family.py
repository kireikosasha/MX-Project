"""Aquarium water clarity.

A tank starts with clean water and its filter pump switched off. The keeper's
actions are read in order. "pump" flips the pump between off and on. "feed"
drops food in: while the pump is off the leftover food dirties the water by
one step (dirt never goes above 3), while the pump is on the food is filtered
and nothing changes. "snail" lets a snail graze, cleaning the water by one step
(dirt never goes below 0). "rinse" does a water change, resetting dirt to 0, and
the keeper also switches the pump off while doing it. The answer is how the
water looks after the last action: dirt 0 is "clear", 1 is "hazy", 2 is
"cloudy" and 3 is "murky".
"""

ALPHABET = ["feed", "pump", "snail", "rinse"]

_LOOKS = ["clear", "hazy", "cloudy", "murky"]


def answer(seq):
    pump_on = False
    dirt = 0
    for tok in seq:
        if tok == "pump":
            pump_on = not pump_on
        elif tok == "feed":
            if not pump_on and dirt < 3:
                dirt += 1
        elif tok == "snail":
            if dirt > 0:
                dirt -= 1
        elif tok == "rinse":
            dirt = 0
            pump_on = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _LOOKS[dirt]
