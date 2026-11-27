"""Planetarium comet sightings.

The dome starts empty with the house lights on. "enter" lets one visitor
into the dome; "leave" lets one visitor out if anyone is inside (if the
dome is empty it does nothing). "dim" switches the house lights off and
"lit" switches them on (repeating the current state changes nothing).
"comet" projects a comet across the dome: it counts as one sighting only
if, at that moment, the lights are off and at least one visitor is inside;
otherwise nobody sees it and it does not count. The answer is the total
number of sightings over the whole sequence, written as a decimal string
(for example "0" or "3").
"""

ALPHABET = ["enter", "leave", "dim", "lit", "comet"]


def answer(seq):
    visitors = 0
    dark = False
    sightings = 0
    for tok in seq:
        if tok == "enter":
            visitors += 1
        elif tok == "leave":
            if visitors > 0:
                visitors -= 1
        elif tok == "dim":
            dark = True
        elif tok == "lit":
            dark = False
        elif tok == "comet":
            if dark and visitors > 0:
                sightings += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sightings)
