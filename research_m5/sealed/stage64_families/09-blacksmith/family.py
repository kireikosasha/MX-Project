"""Blacksmith: the temper of a piece of steel after a session at the forge.

A piece of steel starts cold and soft. Its temper is one of soft < hard <
brittle, and it is either hot or cold. Reading the tokens in order:
"heat" makes cold steel hot, and if it was brittle the heating tempers it
back to hard (heating steel that is already hot does nothing). "rest" lets
hot steel cool slowly: it becomes cold and annealed to soft (resting cold
steel does nothing). "quench" plunges hot steel into water: it becomes cold
and its temper rises one step (soft to hard, hard to brittle, brittle stays
brittle); quenching cold steel does nothing. "hammer" on hot steel only
shapes it and leaves the temper alone, but hammering cold steel work-hardens
it, raising the temper one step (brittle stays brittle). The answer is the
temper at the end: "soft", "hard" or "brittle".
"""

ALPHABET = ["heat", "hammer", "quench", "rest"]

_TEMPERS = ["soft", "hard", "brittle"]


def answer(seq):
    hot = False
    temper = 0  # 0 soft, 1 hard, 2 brittle
    for tok in seq:
        if tok == "heat":
            if not hot:
                hot = True
                if temper == 2:
                    temper = 1
        elif tok == "rest":
            if hot:
                hot = False
                temper = 0
        elif tok == "quench":
            if hot:
                hot = False
                temper = min(temper + 1, 2)
        elif tok == "hammer":
            if not hot:
                temper = min(temper + 1, 2)
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _TEMPERS[temper]
