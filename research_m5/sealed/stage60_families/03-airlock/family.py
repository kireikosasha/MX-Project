"""Airlock control panel log.

The airlock chamber has a pressure gauge with four levels, 0 (vacuum) to 3
(full station pressure), an inner door (to the station) and an outer door (to
space). It starts sealed at pressure 3 with both doors shut. Each token is a
button press, applied in order, with safety interlocks: "pump" raises the
pressure by one (capped at 3) and "vent" lowers it by one (floored at 0), but
both are ignored while either door is open; "inner" opens the inner door only
if both doors are shut and the pressure is 3, otherwise it is ignored;
"outer" opens the outer door only if both doors are shut and the pressure is
0, otherwise it is ignored; "shut" closes whichever door is open (and does
nothing if none is). The answer is the state of the airlock after the last
press: "inner-open", "outer-open", or "sealed-N" where N is the pressure level
(0 to 3) when both doors are shut.
"""

ALPHABET = ["pump", "vent", "inner", "outer", "shut"]


def answer(seq):
    pressure = 3
    door = None  # None, "inner" or "outer"
    for tok in seq:
        if tok == "pump":
            if door is None and pressure < 3:
                pressure += 1
        elif tok == "vent":
            if door is None and pressure > 0:
                pressure -= 1
        elif tok == "inner":
            if door is None and pressure == 3:
                door = "inner"
        elif tok == "outer":
            if door is None and pressure == 0:
                door = "outer"
        elif tok == "shut":
            door = None
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if door is not None:
        return door + "-open"
    return "sealed-%d" % pressure
