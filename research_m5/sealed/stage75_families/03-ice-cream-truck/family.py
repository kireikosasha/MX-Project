"""Ice cream truck: how many scoops are left in the freezer at the end of the day.

The truck starts the day with 3 scoops in its freezer and its jingle switched
off. Read the tokens in order: "honk" flips the jingle (off -> on, on -> off);
"rain" switches the jingle off (it stays off if it already was); "refill" puts
the freezer back to 3 scoops; "kid" is a child on the street, who buys one
scoop only if the jingle is currently on and the freezer still has at least one
scoop (otherwise the kid does not come and nothing changes). The answer is the
number of scoops left after the last token, written as a digit "0" to "3".
"""

ALPHABET = ["kid", "honk", "rain", "refill"]


def answer(seq):
    jingle_on = False
    scoops = 3
    for tok in seq:
        if tok == "honk":
            jingle_on = not jingle_on
        elif tok == "rain":
            jingle_on = False
        elif tok == "refill":
            scoops = 3
        elif tok == "kid":
            if jingle_on and scoops > 0:
                scoops -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(scoops)
