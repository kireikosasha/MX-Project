"""Pizza oven.

A wood-fired oven starts cold, with heat 0, and its heat always stays between
0 and 4. Read the tokens in order: "log" throws wood on the fire and raises the
heat by 1 (it cannot go above 4); "wait" lets the oven cool and lowers the heat
by 1 (it cannot go below 0); "pizza" bakes one pizza at the current heat. A
pizza baked at heat 0 or 1 comes out raw, at heat 2 or 3 it comes out well
baked, and at heat 4 it comes out burnt. After a pizza is taken out, the cold
dough has cooled the oven, so the heat drops by 1 (not below 0). The answer is
the number of well-baked pizzas over the whole sequence, written as a decimal
string (for example "0", "3").
"""

ALPHABET = ["log", "wait", "pizza"]


def answer(seq):
    heat = 0
    good = 0
    for tok in seq:
        if tok == "log":
            if heat < 4:
                heat += 1
        elif tok == "wait":
            if heat > 0:
                heat -= 1
        elif tok == "pizza":
            if 2 <= heat <= 3:
                good += 1
            if heat > 0:
                heat -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(good)
