"""Orchard picker's harvest.

A picker works under the apple trees carrying a basket that holds at most 3 apples,
and starts out in dry weather with an empty basket and an empty cart. Read the tokens
in order: "pick" puts one apple in the basket, unless the basket already holds 3 (then
the apple falls to the ground and is lost) or it is raining (the picker is sheltering
and picks nothing); "cart" tips everything in the basket into the cart and leaves the
basket empty (this works in any weather); "wasp" spoils one apple in the basket, which
is thrown away (nothing happens if the basket is empty); "rain" starts rain and "sun"
ends it (repeating either has no further effect). The answer is the total number of
apples in the cart at the end, written as a decimal string; apples still in the basket
at the end do not count.
"""

ALPHABET = ["pick", "cart", "wasp", "rain", "sun"]

BASKET_SIZE = 3


def answer(seq):
    basket = 0
    cart = 0
    raining = False
    for tok in seq:
        if tok == "pick":
            if not raining and basket < BASKET_SIZE:
                basket += 1
        elif tok == "cart":
            cart += basket
            basket = 0
        elif tok == "wasp":
            if basket > 0:
                basket -= 1
        elif tok == "rain":
            raining = True
        elif tok == "sun":
            raining = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(cart)
