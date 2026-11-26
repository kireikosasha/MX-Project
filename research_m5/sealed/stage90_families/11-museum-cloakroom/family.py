"""Museum cloakroom.

The cloakroom has a coat rail and a bag shelf, both empty when the museum
opens, and the tokens are read in order as events of the day: "coat" means a
visitor hangs one coat on the rail, "bag" means a visitor leaves one bag on the
shelf, and "claim" means a visitor hands in a ticket and the attendant returns
one item, always taking it from whichever place currently holds more items
(from the coat rail when the two hold the same number), or turning the visitor
away with nothing if both are empty. At closing time the answer says what is
left: "coats" if more coats than bags remain, "bags" if more bags than coats
remain, "level" if the same nonzero number of each remains, and "empty" if
nothing at all is left.
"""

ALPHABET = ["coat", "bag", "claim"]


def answer(seq):
    coats = 0
    bags = 0
    for tok in seq:
        if tok == "coat":
            coats += 1
        elif tok == "bag":
            bags += 1
        elif tok == "claim":
            if coats == 0 and bags == 0:
                continue
            if coats >= bags:
                coats -= 1
            else:
                bags -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if coats > bags:
        return "coats"
    if bags > coats:
        return "bags"
    if coats == 0:
        return "empty"
    return "level"
