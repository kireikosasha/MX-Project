"""Printing press: counting finished sheets into gatherings of three.

A small hand press starts with a dry roller (ink level 0) and no sheet on the
bed. "ink" adds one level of ink to the roller, up to a maximum of 3. "wipe"
cleans the roller back to ink level 0. "feed" lays a fresh sheet on the bed (if
a sheet is already there, nothing changes). "press" brings the platen down: if
a sheet is on the bed and the roller has ink level 1 or more, a good sheet is
printed; if a sheet is on the bed but the roller is dry, the sheet comes out
blank and is thrown away; either way the sheet leaves the bed. If there is no
sheet on the bed, nothing is printed. Every "press" (with or without a sheet)
uses up one level of ink if there is any. The binder collects the good sheets
into gatherings of three, and the answer is how many good sheets are in the
gathering still being filled at the end: "empty" (a multiple of three good
sheets, including none), "one" or "two".
"""

ALPHABET = ["ink", "press", "wipe", "feed"]

_NAMES = ("empty", "one", "two")


def answer(seq):
    ink = 0
    sheet = False
    good = 0
    for tok in seq:
        if tok == "ink":
            if ink < 3:
                ink += 1
        elif tok == "wipe":
            ink = 0
        elif tok == "feed":
            sheet = True
        elif tok == "press":
            if sheet:
                if ink > 0:
                    good = (good + 1) % 3
                sheet = False
            if ink > 0:
                ink -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _NAMES[good]
