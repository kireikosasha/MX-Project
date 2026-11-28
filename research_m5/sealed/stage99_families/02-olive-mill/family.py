"""Olive mill: hopper versus shelf.

The mill starts with an empty hopper and an empty oil shelf. Read the tokens in
order. "basket": a farmer tips one basket of olives into the hopper (hopper +1).
"grind": if the hopper holds at least one basket, the millstone grinds one
basket and the press fills one jar of oil, so the hopper loses one basket and
the shelf gains one jar; if the hopper is empty, the stone turns dry and nothing
changes. "buyer": if the shelf holds at least one jar, the buyer carries one jar
away (shelf -1); if the shelf is empty, the buyer leaves empty-handed and
nothing changes. Neither count ever goes below zero and neither has an upper
limit. After the last token, the answer is "olives" if more baskets are waiting
in the hopper than there are jars on the shelf, "oil" if there are more jars on
the shelf than baskets in the hopper, and "even" if the two counts are equal.
"""

ALPHABET = ["basket", "grind", "buyer"]


def answer(seq):
    hopper = 0
    shelf = 0
    for tok in seq:
        if tok == "basket":
            hopper += 1
        elif tok == "grind":
            if hopper > 0:
                hopper -= 1
                shelf += 1
        elif tok == "buyer":
            if shelf > 0:
                shelf -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if hopper > shelf:
        return "olives"
    if shelf > hopper:
        return "oil"
    return "even"
