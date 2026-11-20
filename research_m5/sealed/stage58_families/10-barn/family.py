"""Barn hayloft.

The barn starts with an empty hayloft (0 bales) and dry weather. Read the
tokens in order: "rain" makes the weather wet and "sun" makes it dry; "hay" is
a delivered bale, which is stacked in the loft only if the weather is dry
(a bale delivered in the wet is spoiled and thrown away) and only if the loft
holds fewer than 5 bales (a full loft turns the bale away); "cow" means a cow
comes to feed and eats one bale from the loft if there is any (otherwise
nothing happens). The answer is the number of bales left in the loft after the
last token, written as a digit from "0" to "5".
"""

ALPHABET = ["hay", "cow", "rain", "sun"]

CAPACITY = 5


def answer(seq):
    bales = 0
    wet = False
    for tok in seq:
        if tok == "rain":
            wet = True
        elif tok == "sun":
            wet = False
        elif tok == "hay":
            if not wet and bales < CAPACITY:
                bales += 1
        elif tok == "cow":
            if bales > 0:
                bales -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(bales)
