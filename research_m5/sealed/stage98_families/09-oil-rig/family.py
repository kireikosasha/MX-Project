"""Offshore oil rig storage tank.

The rig starts in calm weather with an empty storage tank, and the tokens are
read in order. 'storm' puts the rig into storm mode and 'calm' returns it to
calm weather (repeating either one changes nothing). 'pump' adds one barrel to
the tank, but only in calm weather: during a storm the wells are shut in and
nothing is pumped. 'tanker' takes one barrel out of the tank, but only in calm
weather and only if the tank is not empty: during a storm the tanker cannot
moor, and with an empty tank it leaves with nothing; either way the tanker is
gone and does not come back. The answer is the number of barrels in the tank
after the last token, written as a decimal string such as '0' or '3'.
"""

ALPHABET = ["pump", "tanker", "storm", "calm"]


def answer(seq):
    stormy = False
    tank = 0
    for tok in seq:
        if tok == "storm":
            stormy = True
        elif tok == "calm":
            stormy = False
        elif tok == "pump":
            if not stormy:
                tank += 1
        elif tok == "tanker":
            if not stormy and tank > 0:
                tank -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(tank)
