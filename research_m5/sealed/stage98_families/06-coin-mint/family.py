"""A coin mint press with a hopper of blanks and a two-design die.

The mint starts with an empty hopper, the die showing the "eagle" design, and
no coins struck. Each token is one event: "blank" drops one blank disc into the
hopper (the hopper can hold any number); "swap" changes the die to the other
design (eagle becomes crown, crown becomes eagle); "strike" brings the press
down: if the hopper holds at least one blank, one blank is used up and becomes
a coin bearing the design the die shows at that moment, but if the hopper is
empty the press strikes nothing and no coin is made. After the whole sequence
the answer says which design is on more of the struck coins: "eagle", "crown",
or "even" when both designs have the same number of coins (including when no
coin was struck at all).
"""

ALPHABET = ["blank", "strike", "swap"]


def answer(seq):
    hopper = 0          # blanks waiting in the hopper (unbounded)
    design = "eagle"    # design currently on the die
    lead = 0            # eagle coins minus crown coins (unbounded)
    for tok in seq:
        if tok == "blank":
            hopper += 1
        elif tok == "swap":
            design = "crown" if design == "eagle" else "eagle"
        elif tok == "strike":
            if hopper > 0:
                hopper -= 1
                lead += 1 if design == "eagle" else -1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if lead > 0:
        return "eagle"
    if lead < 0:
        return "crown"
    return "even"
