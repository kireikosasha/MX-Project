"""Water tower with a float switch.

A town water tower holds 0 to 8 units of water; it starts at 4 units with its
supply pump switched on.  Tokens are read left to right: "rain" adds 1 unit,
"pump" adds 2 units but only while the pump is switched on (otherwise it does
nothing), "tap" removes 1 unit, and "hydrant" removes 3 units.  The level never
goes above 8 (extra water spills over) or below 0.  After every token the float
switch is checked: if the tank is full (8) the pump is switched off, and if the
level is 2 or lower the pump is switched back on; between those levels the pump
keeps whatever state it had.  The answer is the final level and the final pump
state, written as "<level>-on" or "<level>-off" (for example "5-off").
"""

ALPHABET = ["rain", "pump", "tap", "hydrant"]

CAPACITY = 8
START_LEVEL = 4
LOW_MARK = 2


def answer(seq):
    level = START_LEVEL
    pump_on = True
    for tok in seq:
        if tok == "rain":
            level += 1
        elif tok == "pump":
            if pump_on:
                level += 2
        elif tok == "tap":
            level -= 1
        elif tok == "hydrant":
            level -= 3
        else:
            raise ValueError("unknown token: %r" % (tok,))
        if level > CAPACITY:
            level = CAPACITY
        if level < 0:
            level = 0
        if level == CAPACITY:
            pump_on = False
        elif level <= LOW_MARK:
            pump_on = True
    return "%d-%s" % (level, "on" if pump_on else "off")
