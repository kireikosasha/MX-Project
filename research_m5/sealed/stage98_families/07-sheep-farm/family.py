"""Sheep farm fold.

A farmer keeps a fold with one gate; the gate starts closed, the fold starts
empty and no sheep are waiting outside. Read the tokens in order. "gate"
swings the gate: closed becomes open, open becomes closed. "ewe" means a ewe
arrives: if the gate is open she walks straight into the fold, otherwise she
joins the sheep waiting outside. "dog" means the sheepdog runs a round: if the
gate is open it drives every waiting sheep into the fold (none are left
outside); if the gate is closed it does nothing. "wolf" means a wolf strikes:
if the gate is open it takes one sheep from the fold, otherwise it takes one
of the sheep waiting outside; if there is no sheep in that place it leaves
empty-handed. The answer is the number of sheep in the fold after the last
token, written as a decimal number (for example "0" or "7").
"""

ALPHABET = ["ewe", "gate", "dog", "wolf"]


def answer(seq):
    gate_open = False
    fold = 0
    outside = 0
    for tok in seq:
        if tok == "gate":
            gate_open = not gate_open
        elif tok == "ewe":
            if gate_open:
                fold += 1
            else:
                outside += 1
        elif tok == "dog":
            if gate_open:
                fold += outside
                outside = 0
        elif tok == "wolf":
            if gate_open:
                if fold > 0:
                    fold -= 1
            else:
                if outside > 0:
                    outside -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(fold)
