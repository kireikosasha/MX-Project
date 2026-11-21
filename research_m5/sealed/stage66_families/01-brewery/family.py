"""Brewery fermenting tank.

A single tank starts empty and holds two quantities, sugar and alcohol, each an
integer from 0 to 3. Reading the tokens in order: "malt" adds one unit of sugar
(nothing happens if sugar is already 3); "yeast" turns one unit of sugar into one
unit of alcohol, but only if there is some sugar and alcohol is below 3
(otherwise nothing happens); "water" dilutes the tank, lowering sugar by one and
alcohol by one (each separately, never below 0); "drain" empties the tank,
setting both to 0. After the last token the answer tastes the tank: "empty" if
sugar and alcohol are both 0, "sweet" if sugar is greater than alcohol,
"strong" if alcohol is greater than sugar, and "balanced" if they are equal and
not zero.
"""

ALPHABET = ["malt", "yeast", "water", "drain"]

CAP = 3


def answer(seq):
    sugar = 0
    alcohol = 0
    for tok in seq:
        if tok == "malt":
            if sugar < CAP:
                sugar += 1
        elif tok == "yeast":
            if sugar > 0 and alcohol < CAP:
                sugar -= 1
                alcohol += 1
        elif tok == "water":
            if sugar > 0:
                sugar -= 1
            if alcohol > 0:
                alcohol -= 1
        elif tok == "drain":
            sugar = 0
            alcohol = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if sugar == 0 and alcohol == 0:
        return "empty"
    if sugar > alcohol:
        return "sweet"
    if alcohol > sugar:
        return "strong"
    return "balanced"
