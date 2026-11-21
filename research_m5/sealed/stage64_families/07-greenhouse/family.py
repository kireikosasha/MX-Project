"""Greenhouse seedling.

A single plant grows in a greenhouse whose air is "cold", "mild" or "hot";
the air starts mild and the plant starts as a "seed". Read the tokens in
order. "sun" makes the air one step warmer (cold -> mild -> hot, and hot
stays hot); "shade" makes it one step cooler (hot -> mild -> cold, and cold
stays cold); "vent" sets the air straight to mild. "water" acts on the plant
according to the air at that moment: in mild air the plant grows one stage
along the cycle seed -> sprout -> bud -> bloom -> seed (a bloom drops its
seed and the cycle starts over); in cold air the soaked roots rot and the
plant goes back to "seed" whatever stage it had; in hot air the water
evaporates and nothing happens. The answer is the plant's stage after the
last token: "seed", "sprout", "bud" or "bloom".
"""

ALPHABET = ["sun", "shade", "vent", "water"]

_STAGES = ["seed", "sprout", "bud", "bloom"]


def answer(seq):
    temp = 1   # 0 = cold, 1 = mild, 2 = hot
    stage = 0  # index into _STAGES
    for tok in seq:
        if tok == "sun":
            if temp < 2:
                temp += 1
        elif tok == "shade":
            if temp > 0:
                temp -= 1
        elif tok == "vent":
            temp = 1
        elif tok == "water":
            if temp == 1:
                stage = (stage + 1) % 4
            elif temp == 0:
                stage = 0
            # hot: evaporates, nothing happens
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _STAGES[stage]
