"""Food bank shelf: count the families turned away.

A food bank starts with an empty shelf and reads the day's events in order.
"can" puts one can on the shelf and "loaf" puts one loaf on the shelf.
"family" means a family arrives and is handed one loaf if any loaf is on the
shelf and, separately, one can if any can is on the shelf; if the shelf holds
neither a loaf nor a can at that moment, the family leaves with nothing and is
counted as turned away. "night" means the day ends: every loaf still on the
shelf has gone stale and is thrown out, while cans keep and stay on the shelf.
The answer is the total number of families turned away over the whole
sequence, written as a decimal number (for example "0" or "3").
"""

ALPHABET = ["can", "loaf", "family", "night"]


def answer(seq):
    cans = 0
    loaves = 0
    turned_away = 0
    for token in seq:
        if token == "can":
            cans += 1
        elif token == "loaf":
            loaves += 1
        elif token == "family":
            if loaves == 0 and cans == 0:
                turned_away += 1
            else:
                if loaves > 0:
                    loaves -= 1
                if cans > 0:
                    cans -= 1
        elif token == "night":
            loaves = 0
        else:
            raise ValueError("unknown token: %r" % (token,))
    return str(turned_away)
