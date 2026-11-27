"""Sushi conveyor belt.

The belt starts empty and the diner starts out craving salmon. Read the
tokens in order: "salmon" or "tuna" means the chef puts one plate of that
fish on the belt; "switch" means the diner's craving flips (salmon to tuna,
or tuna to salmon); "grab" means the diner takes one plate off the belt --
a plate of the fish they currently crave if there is one, otherwise a plate
of the other fish if there is one, and nothing at all if the belt is empty.
After the last token, look at the plates still on the belt: the answer is
"salmon" if more salmon plates than tuna plates remain, "tuna" if more tuna
plates remain, and "even" if the two numbers are equal (including an empty
belt).
"""

ALPHABET = ["salmon", "tuna", "grab", "switch"]


def answer(seq):
    salmon = 0
    tuna = 0
    craving_salmon = True
    for tok in seq:
        if tok == "salmon":
            salmon += 1
        elif tok == "tuna":
            tuna += 1
        elif tok == "switch":
            craving_salmon = not craving_salmon
        elif tok == "grab":
            if craving_salmon:
                if salmon > 0:
                    salmon -= 1
                elif tuna > 0:
                    tuna -= 1
            else:
                if tuna > 0:
                    tuna -= 1
                elif salmon > 0:
                    salmon -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if salmon > tuna:
        return "salmon"
    if tuna > salmon:
        return "tuna"
    return "even"
