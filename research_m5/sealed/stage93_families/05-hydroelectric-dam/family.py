"""Hydroelectric dam: the reservoir starts empty and no turbine is running.
Read the tokens in order. "rain" adds one unit of water to the reservoir
(the reservoir has no upper limit). "open" starts one more turbine, but the
dam has only two turbines, so it does nothing when both already run. "shut"
stops one running turbine, and does nothing when none is running. "hour"
lets one hour pass: each running turbine takes one unit of water from the
reservoir, if any water is left, and turns it into one megawatt-hour of
electricity (so the hour produces min(running turbines, water units) and
the reservoir drops by that much). The answer is the total number of
megawatt-hours produced over the whole sequence, written as a decimal
integer (e.g. "0", "3", "17").
"""

ALPHABET = ["rain", "open", "shut", "hour"]

MAX_TURBINES = 2


def answer(seq):
    water = 0
    running = 0
    energy = 0
    for tok in seq:
        if tok == "rain":
            water += 1
        elif tok == "open":
            if running < MAX_TURBINES:
                running += 1
        elif tok == "shut":
            if running > 0:
                running -= 1
        elif tok == "hour":
            used = min(running, water)
            water -= used
            energy += used
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(energy)
