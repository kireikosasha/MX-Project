"""A dumpling stall at a night market.

The stall's steamer holds at most 3 baskets of dumplings and starts empty, and
the evening starts dry. Read the tokens in order: "steam" puts one more basket
in the steamer (nothing happens if it already holds 3); "buy" is a passer-by
who takes one basket if there is any (nothing happens if the steamer is empty),
but while it is raining nobody stops at the stall, so "buy" is ignored; "rain"
makes it start raining (or keep raining) and "clear" makes it dry again. The
answer is the number of baskets left in the steamer when the market closes,
written as a digit "0", "1", "2" or "3".
"""

ALPHABET = ["steam", "buy", "rain", "clear"]


def answer(seq):
    baskets = 0
    raining = False
    for tok in seq:
        if tok == "steam":
            if baskets < 3:
                baskets += 1
        elif tok == "buy":
            if not raining and baskets > 0:
                baskets -= 1
        elif tok == "rain":
            raining = True
        elif tok == "clear":
            raining = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(baskets)
