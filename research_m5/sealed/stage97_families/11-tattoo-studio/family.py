"""A one-artist tattoo studio over one day; the answer says whether more customers got inked or walked out.

The studio opens with an empty waiting bench and one stencil ready. Read the
tokens in order. "walkin": one customer sits down on the bench (the bench has
no size limit). "stencil": the artist prepares one more stencil, but at most
3 stencils can be ready at once, so a fourth is not made. "tattoo": if at least
one customer is on the bench and at least one stencil is ready, one customer is
tattooed and leaves inked, using up one stencil; otherwise nothing happens.
"leave": if at least one customer is on the bench, one of them gives up and
walks out unhappy; with an empty bench nothing happens. At the end, the answer
is "inked" if more customers were tattooed than walked out, "lost" if more
walked out than were tattooed, and "even" if the two numbers are equal
(including when both are zero).
"""

ALPHABET = ["walkin", "stencil", "tattoo", "leave"]

MAX_STENCILS = 3


def answer(seq):
    bench = 0
    stencils = 1
    inked = 0
    lost = 0
    for tok in seq:
        if tok == "walkin":
            bench += 1
        elif tok == "stencil":
            if stencils < MAX_STENCILS:
                stencils += 1
        elif tok == "tattoo":
            if bench > 0 and stencils > 0:
                bench -= 1
                stencils -= 1
                inked += 1
        elif tok == "leave":
            if bench > 0:
                bench -= 1
                lost += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if inked > lost:
        return "inked"
    if lost > inked:
        return "lost"
    return "even"
