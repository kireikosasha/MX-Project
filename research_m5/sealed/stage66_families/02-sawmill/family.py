"""Sawmill blade and the two piles.

The mill's saw blade starts fully sharp, with sharpness 3. The tokens are read
in order. A "log" is a plain log: if the blade's sharpness is at least 1 it is
sawn into a board and the blade loses 1 sharpness; otherwise it is set aside on
the reject pile and the blade is unchanged. A "knot" is a knotty log: it needs
sharpness at least 2, and if so it is sawn into a board and the blade loses 2
sharpness; otherwise it goes to the reject pile and the blade is unchanged.
"hone" puts the blade back to full sharpness 3. "rest" is an idle moment and
changes nothing. At the end, the answer compares the board pile with the reject
pile: "boards" if there are more boards than rejects, "rejects" if there are
more rejects than boards, and "even" if the two piles are the same size.
"""

ALPHABET = ["log", "knot", "hone", "rest"]

FULL = 3


def answer(seq):
    sharp = FULL
    boards = 0
    rejects = 0
    for tok in seq:
        if tok == "log":
            if sharp >= 1:
                sharp -= 1
                boards += 1
            else:
                rejects += 1
        elif tok == "knot":
            if sharp >= 2:
                sharp -= 2
                boards += 1
            else:
                rejects += 1
        elif tok == "hone":
            sharp = FULL
        elif tok == "rest":
            pass
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if boards > rejects:
        return "boards"
    if rejects > boards:
        return "rejects"
    return "even"
