"""Pot on the stove.

A pot starts cold with its lid off, and the tokens are things the cook does to
it. The pot's temperature is one of four levels, cold < warm < simmer < boil.
"lid" puts the lid on if it is off and takes it off if it is on. "heat" raises
the temperature by one level with the lid off, or by two levels with the lid on,
never above boil. "water" pours in cold water and lowers the temperature by two
levels (never below cold), but only when the lid is off; with the lid on nothing
happens. "wait" lets the pot lose heat by one level (never below cold) when the
lid is off; with the lid on the heat is kept and nothing happens. The answer is
the name of the pot's temperature level after all tokens have been read.
"""

ALPHABET = ["heat", "water", "wait", "lid"]

_LEVELS = ["cold", "warm", "simmer", "boil"]


def answer(seq):
    temp = 0
    lid = False
    for tok in seq:
        if tok == "lid":
            lid = not lid
        elif tok == "heat":
            temp = min(3, temp + (2 if lid else 1))
        elif tok == "water":
            if not lid:
                temp = max(0, temp - 2)
        elif tok == "wait":
            if not lid:
                temp = max(0, temp - 1)
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _LEVELS[temp]
