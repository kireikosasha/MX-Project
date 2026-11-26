"""Bike-sharing station: count the riders who leave without a bike.

The station opens with no bikes in its docks. Read the tokens in order:
"back" means a rider docks a working bike (working bikes +1); "flat" means a
rider docks a bike with a flat tyre, which stays in the dock but cannot be
ridden (broken bikes +1); "take" means a rider wants a bike: if there is at
least one working bike they ride off on it (working bikes -1), otherwise they
are turned away (turned-away count +1), even if broken bikes are sitting in
the docks; "van" means the repair van visits and fixes every broken bike in
the station, so all broken bikes become working bikes (working += broken,
broken = 0). The answer is the total number of riders turned away over the
whole sequence, written as a decimal string such as "0" or "3".
"""

ALPHABET = ["back", "flat", "take", "van"]


def answer(seq):
    working = 0
    broken = 0
    turned_away = 0
    for tok in seq:
        if tok == "back":
            working += 1
        elif tok == "flat":
            broken += 1
        elif tok == "take":
            if working > 0:
                working -= 1
            else:
                turned_away += 1
        elif tok == "van":
            working += broken
            broken = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(turned_away)
