"""Garden bed: one plot is followed through a season of events.

The plot starts bare with a dry soil (water 0). Stages, in order, are
bare, sprout, leafy, bloom, fruit. "seed" turns a bare plot into a sprout
(no effect otherwise). "rain" adds one unit of water to the soil, which holds
at most 2. "sun" does nothing to a bare plot; otherwise, if the soil has
water, one unit is used up and the plant grows one stage (fruit stays fruit),
and if the soil is dry the plant wilts back one stage (a sprout dies back to
bare). "frost" knocks bloom or fruit back to leafy and kills a sprout (back to
bare); leafy plants and bare plots are unaffected. "pick" takes a bloom or
fruit back to leafy; otherwise nothing. Water is never changed by seed, frost
or pick. The answer is the name of the stage after the whole sequence.
"""

ALPHABET = ["seed", "rain", "sun", "frost", "pick"]

_STAGES = ["bare", "sprout", "leafy", "bloom", "fruit"]


def answer(seq):
    stage = 0
    water = 0
    for tok in seq:
        if tok == "seed":
            if stage == 0:
                stage = 1
        elif tok == "rain":
            if water < 2:
                water += 1
        elif tok == "sun":
            if stage != 0:
                if water > 0:
                    water -= 1
                    if stage < 4:
                        stage += 1
                else:
                    stage -= 1
        elif tok == "frost":
            if stage >= 3:
                stage = 2
            elif stage == 1:
                stage = 0
        elif tok == "pick":
            if stage >= 3:
                stage = 2
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _STAGES[stage]
