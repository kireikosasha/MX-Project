"""Kayak rental on a lake.

The dock owns 2 kayaks, both tied up at the dock when the day opens and nobody
is waiting. Read the tokens in order. "guest": a guest arrives wanting a
kayak; if a kayak is at the dock they take it and paddle out, otherwise they
join the end of the waiting line. "back": a paddler returns; if anyone is
waiting, the first guest in line takes that kayak straight out again,
otherwise the kayak is tied up at the dock; if no kayak is out on the water,
"back" changes nothing. "rain": a shower passes; every guest currently waiting
in line gives up and goes home without paddling (paddlers already on the water
stay out). At closing time (after the last token) anyone still waiting in line
also goes home without paddling. The answer is the total number of guests who
went home without ever paddling, written as a decimal number (e.g. "0", "3").
"""

ALPHABET = ["guest", "back", "rain"]

FLEET = 2


def answer(seq):
    out = 0          # kayaks currently on the water (0..FLEET)
    line = 0         # guests waiting in line (unbounded)
    went_home = 0    # guests who left without paddling (unbounded)
    for tok in seq:
        if tok == "guest":
            if out < FLEET:
                out += 1
            else:
                line += 1
        elif tok == "back":
            if line > 0:
                line -= 1
            elif out > 0:
                out -= 1
        elif tok == "rain":
            went_home += line
            line = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    went_home += line
    return str(went_home)
