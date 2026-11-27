"""Riding stable: horses come and go, hay piles up in the loft.

The stable starts empty, with no hay. Reading the tokens in order:
"bay" means a bay horse arrives and takes a stall; "grey" means a grey horse
arrives and takes a stall; "hay" means one bale of hay is delivered to the
loft; "ride" means a rider comes and leaves for good with one horse, always
choosing a horse of the colour that currently has more horses in the stalls
(a bay when both colours are equally many, and no horse at all when the
stalls are empty). Hay is never used up. At the end the answer is "empty" if
no horse is left in the stalls; otherwise it is the colour with more horses
in the stalls ("bay", "grey", or "even" when the counts are equal), a dash,
and "fed" if there are at least as many bales as horses in the stalls or
"short" if there are fewer bales than horses (for example "grey-short").
"""

ALPHABET = ["bay", "grey", "ride", "hay"]


def answer(seq):
    bays = 0
    greys = 0
    hay = 0
    for tok in seq:
        if tok == "bay":
            bays += 1
        elif tok == "grey":
            greys += 1
        elif tok == "hay":
            hay += 1
        elif tok == "ride":
            if bays + greys == 0:
                pass
            elif bays >= greys:
                bays -= 1
            else:
                greys -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    horses = bays + greys
    if horses == 0:
        return "empty"
    if bays > greys:
        colour = "bay"
    elif greys > bays:
        colour = "grey"
    else:
        colour = "even"
    return colour + "-" + ("fed" if hay >= horses else "short")
