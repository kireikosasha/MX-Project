"""Bell foundry.

The foundry starts with no molten bronze, no empty moulds and no bells. Read
the tokens in order: "melt" adds one measure of molten bronze to the furnace;
"mould" readies one more empty mould; "pour" casts one bell if there is at
least one measure of bronze and at least one empty mould (using up one of
each), and does nothing otherwise; "crack" means one cast bell is found
cracked: if there is at least one bell, that bell is removed and broken up and
its measure of bronze goes back into the furnace (its mould is not recovered),
and if there is no bell it does nothing. The answer is the number of bells
standing at the end, written as a decimal number (for example "0" or "3").
"""

ALPHABET = ["melt", "mould", "pour", "crack"]


def answer(seq):
    bronze = 0
    moulds = 0
    bells = 0
    for tok in seq:
        if tok == "melt":
            bronze += 1
        elif tok == "mould":
            moulds += 1
        elif tok == "pour":
            if bronze > 0 and moulds > 0:
                bronze -= 1
                moulds -= 1
                bells += 1
        elif tok == "crack":
            if bells > 0:
                bells -= 1
                bronze += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(bells)
