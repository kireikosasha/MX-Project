"""Single-bay car wash: count the cars that drive out spotless.

The wash has one bay and a waiting line. "car" means a car arrives: it goes
straight into the bay if the bay is empty, otherwise it joins the end of the
line. "soap" lathers the car in the bay (whatever state it was in, it is now
soapy). "rinse" turns a soapy car into a rinsed car; rinsing a car that is not
soapy changes nothing. "dry" runs the blower, which always pushes the car in
the bay out of the exit: it counts as spotless only if it was rinsed (a soapy
car leaves streaked, an unwashed car leaves dirty), and then the next car in
line, if any, rolls into the bay unwashed. Any of "soap", "rinse" or "dry" with
an empty bay does nothing. The answer is the number of spotless cars that have
left by the end of the sequence, written as a decimal string.
"""

ALPHABET = ["car", "soap", "rinse", "dry"]


def answer(seq):
    waiting = 0          # cars in line outside the bay
    bay = None           # None, "dirty", "soapy" or "rinsed"
    spotless = 0
    for tok in seq:
        if tok == "car":
            if bay is None:
                bay = "dirty"
            else:
                waiting += 1
        elif tok == "soap":
            if bay is not None:
                bay = "soapy"
        elif tok == "rinse":
            if bay == "soapy":
                bay = "rinsed"
        elif tok == "dry":
            if bay is not None:
                if bay == "rinsed":
                    spotless += 1
                if waiting > 0:
                    waiting -= 1
                    bay = "dirty"
                else:
                    bay = None
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(spotless)
