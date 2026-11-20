"""School canteen hot counter.

The serving counter starts empty and holds at most 4 portions of shepherd's
pie. Read the tokens in order: "cook" brings out a fresh tray of 2 portions,
but anything beyond the counter's 4-portion limit is sent back to the kitchen;
"pupil" takes 1 portion if there is at least one (otherwise leaves empty-handed
and nothing changes); "teacher" takes 2 portions only if at least 2 are on the
counter, and otherwise takes nothing; "spill" means a tray is knocked over and
the counter is cleared to 0. The answer is the number of portions left on the
counter after the last token, written as a digit "0" to "4".
"""

ALPHABET = ["cook", "pupil", "teacher", "spill"]

CAPACITY = 4


def answer(seq):
    portions = 0
    for tok in seq:
        if tok == "cook":
            portions = min(CAPACITY, portions + 2)
        elif tok == "pupil":
            if portions >= 1:
                portions -= 1
        elif tok == "teacher":
            if portions >= 2:
                portions -= 2
        elif tok == "spill":
            portions = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(portions)
