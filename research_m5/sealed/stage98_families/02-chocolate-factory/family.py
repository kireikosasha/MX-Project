"""Chocolate factory tempering line.

The tempering kettle has four temperature levels: cold (0), cool (1), warm (2)
and hot (3); it starts cold. "heat" raises the level by one (it cannot go above
hot) and "chill" lowers it by one (it cannot go below cold). "pour" fills one
mould from the kettle: if the kettle is cool or warm (levels 1 or 2) the
chocolate is properly tempered and the bar comes out glossy, otherwise (cold or
hot) the bar comes out bloomed. The answer compares the total number of glossy
bars with the total number of bloomed bars made over the whole sequence:
"glossy" if there are more glossy bars, "bloomed" if there are more bloomed
bars, and "tie" if the counts are equal (including when nothing was poured).
"""

ALPHABET = ["heat", "chill", "pour"]


def answer(seq):
    level = 0
    balance = 0  # glossy bars minus bloomed bars
    for tok in seq:
        if tok == "heat":
            if level < 3:
                level += 1
        elif tok == "chill":
            if level > 0:
                level -= 1
        elif tok == "pour":
            if level == 1 or level == 2:
                balance += 1
            else:
                balance -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "glossy"
    if balance < 0:
        return "bloomed"
    return "tie"
