"""Coffee roastery: one day at a small roastery, read event by event.

The roastery starts the day with no green beans, no roasted bags on the
shelf and no customers waiting. "sack" means a sack of green beans is
delivered (green stock goes up by one). "roast" means the roaster turns one
green sack into one bag of roasted coffee, but only if there is at least one
green sack in stock; with no green beans the roast does nothing. A freshly
roasted bag goes straight to a waiting customer if anyone is waiting
(that customer leaves served), otherwise it is put on the shelf. "order"
means a customer wants a bag: they take one from the shelf if the shelf is
not empty, otherwise they join the waiting line. "taste" means the roaster
takes one bag from the shelf for a cupping session if the shelf is not
empty (it never takes a bag meant for a waiting customer); with an empty
shelf it does nothing. The answer describes the end of the day: "short N"
if N customers are still waiting, "spare N" if N bags are left on the
shelf, or "even" if nobody is waiting and the shelf is empty.
"""

ALPHABET = ["sack", "roast", "order", "taste"]


def answer(seq):
    green = 0
    shelf = 0
    waiting = 0
    for tok in seq:
        if tok == "sack":
            green += 1
        elif tok == "roast":
            if green > 0:
                green -= 1
                if waiting > 0:
                    waiting -= 1
                else:
                    shelf += 1
        elif tok == "order":
            if shelf > 0:
                shelf -= 1
            else:
                waiting += 1
        elif tok == "taste":
            if shelf > 0:
                shelf -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if waiting > 0:
        return "short %d" % waiting
    if shelf > 0:
        return "spare %d" % shelf
    return "even"
