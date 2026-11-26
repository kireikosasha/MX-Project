"""A small blood bank keeps bags of type O and type A blood; it starts empty and
reads the day's events one by one. "giveO" and "giveA" mean a donor brings one
bag of that type, which goes on the shelf. "needO" means a type-O patient
arrives: that patient can only take an O bag, so one O bag is used if there is
any, otherwise the patient is turned away. "needA" means a type-A patient
arrives: an A bag is used if there is any, otherwise an O bag is used (O can be
given to anyone), and if neither is on the shelf the patient is turned away.
The answer is the total number of patients turned away during the whole day,
written as a decimal number (for example "0" or "3").
"""

ALPHABET = ["giveO", "giveA", "needO", "needA"]


def answer(seq):
    o_bags = 0
    a_bags = 0
    turned_away = 0
    for tok in seq:
        if tok == "giveO":
            o_bags += 1
        elif tok == "giveA":
            a_bags += 1
        elif tok == "needO":
            if o_bags > 0:
                o_bags -= 1
            else:
                turned_away += 1
        elif tok == "needA":
            if a_bags > 0:
                a_bags -= 1
            elif o_bags > 0:
                o_bags -= 1
            else:
                turned_away += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(turned_away)
