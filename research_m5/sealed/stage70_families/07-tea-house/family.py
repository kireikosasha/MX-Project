"""Tea-house evening.

The tea master keeps one pot that holds at most 3 cups (empty at opening) and
a line of waiting guests (empty at opening). Reading the evening token by
token: "guest" means a new guest arrives and joins the waiting line; "brew"
fills the pot to exactly 3 cups, whatever was left in it; "pour" does nothing
if the pot is empty, otherwise one cup leaves the pot and, if anyone is
waiting, one waiting guest is served (and leaves the line), while if nobody is
waiting that cup is simply wasted; "leave" means one waiting guest gives up
and walks out unserved (nothing happens if nobody is waiting). At closing
time, every guest still in line also counts as unserved. The answer is the
house's reputation for the evening: "praised" if more guests were served than
went unserved, "scolded" if more went unserved than were served, and "even"
if the two counts are equal (including an evening with no guests at all).
"""

ALPHABET = ["guest", "brew", "pour", "leave"]

POT_SIZE = 3


def answer(seq):
    pot = 0
    waiting = 0
    served = 0
    unserved = 0
    for tok in seq:
        if tok == "guest":
            waiting += 1
        elif tok == "brew":
            pot = POT_SIZE
        elif tok == "pour":
            if pot > 0:
                pot -= 1
                if waiting > 0:
                    waiting -= 1
                    served += 1
        elif tok == "leave":
            if waiting > 0:
                waiting -= 1
                unserved += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    unserved += waiting
    if served > unserved:
        return "praised"
    if unserved > served:
        return "scolded"
    return "even"
