"""Car rental desk.

The lot starts empty and each token is one event at the desk, read in order:
"small" means a small car is returned and parked on the lot, "large" means a
large car is returned and parked on the lot, "solo" means a single traveller
arrives and drives off in a small car if one is on the lot, otherwise in a
large car if one is on the lot, otherwise is turned away, and "group" means a
group arrives and drives off in a large car if one is on the lot, otherwise is
turned away (a group never takes a small car). The answer is the total number
of customers (solo travellers and groups together) turned away over the whole
sequence, written as a decimal string such as "0" or "3".
"""

ALPHABET = ["small", "large", "solo", "group"]


def answer(seq):
    small = 0
    large = 0
    turned_away = 0
    for tok in seq:
        if tok == "small":
            small += 1
        elif tok == "large":
            large += 1
        elif tok == "solo":
            if small > 0:
                small -= 1
            elif large > 0:
                large -= 1
            else:
                turned_away += 1
        elif tok == "group":
            if large > 0:
                large -= 1
            else:
                turned_away += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(turned_away)
