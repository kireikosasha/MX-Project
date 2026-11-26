"""Storm shelter: cots versus people.

An emergency shelter opens with no cots and nobody inside, and the tokens are
read in order. "cot" means volunteers set up one more cot. "arrive" means one
more person comes in. "leave" means one person goes home (if nobody is inside,
nothing happens). "lull" means the storm eases for a moment and every person
who does not have a cot (the people beyond the number of cots) goes home, while
everyone who has a cot stays. Cots are never taken away, and people are
interchangeable, so all that matters is how many cots there are and how many
people are inside. The answer compares the two at the end: "N waiting" if there
are N more people than cots, "N free" if there are N more cots than people,
and "full" if they are equal.
"""

ALPHABET = ["arrive", "leave", "cot", "lull"]


def answer(seq):
    cots = 0
    people = 0
    for tok in seq:
        if tok == "arrive":
            people += 1
        elif tok == "leave":
            if people > 0:
                people -= 1
        elif tok == "cot":
            cots += 1
        elif tok == "lull":
            if people > cots:
                people = cots
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if people > cots:
        return "%d waiting" % (people - cots)
    if cots > people:
        return "%d free" % (cots - people)
    return "full"
