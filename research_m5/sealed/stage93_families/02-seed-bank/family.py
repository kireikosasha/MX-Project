"""Seed bank stock count.

A seed bank keeps packets of two crops, rye and flax. It starts empty, and the
season starts as rye season. Read the tokens in order: "rye" or "flax"
deposits one packet of that crop; "turn" switches the season (rye season
becomes flax season and back again); "ask" is a grower asking for one packet
of the crop that is in season right now. If the bank has a packet of that
crop, one is handed out and the stock goes down by one. If not, the grower
leaves with nothing and nothing changes (the grower never takes the other
crop). After the last token, the answer is "empty" if the bank holds no
packets at all, "even" if it holds the same nonzero number of rye and flax
packets, and otherwise the name of the crop it holds more packets of ("rye"
or "flax").
"""

ALPHABET = ["rye", "flax", "ask", "turn"]


def answer(seq):
    stock = {"rye": 0, "flax": 0}
    season = "rye"
    for tok in seq:
        if tok == "rye" or tok == "flax":
            stock[tok] += 1
        elif tok == "turn":
            season = "flax" if season == "rye" else "rye"
        elif tok == "ask":
            if stock[season] > 0:
                stock[season] -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    rye, flax = stock["rye"], stock["flax"]
    if rye == 0 and flax == 0:
        return "empty"
    if rye == flax:
        return "even"
    return "rye" if rye > flax else "flax"
