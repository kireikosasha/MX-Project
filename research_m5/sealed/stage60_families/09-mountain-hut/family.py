"""Mountain hut stove log.

A mountain hut starts empty, with no guests and no firewood stacked by the stove.
Reading the tokens in order: "hiker" means one hiker arrives and stays; "leave"
means one guest departs (nothing happens if the hut is already empty); "log" means
one log is added to the woodpile, which holds at most 3 logs (extra logs are
wasted); "dusk" means a night falls: if nobody is in the hut, the stove is not
lit and nothing happens; if there are guests and at least one log, one log is
burned and the night is warm; if there are guests but no wood, it is a cold
night. The answer is the total number of cold nights, written as a decimal
string (e.g. "0", "2").
"""

ALPHABET = ["hiker", "leave", "log", "dusk"]

WOODPILE_CAPACITY = 3


def answer(seq):
    guests = 0
    wood = 0
    cold_nights = 0
    for tok in seq:
        if tok == "hiker":
            guests += 1
        elif tok == "leave":
            if guests > 0:
                guests -= 1
        elif tok == "log":
            if wood < WOODPILE_CAPACITY:
                wood += 1
        elif tok == "dusk":
            if guests > 0:
                if wood > 0:
                    wood -= 1
                else:
                    cold_nights += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(cold_nights)
