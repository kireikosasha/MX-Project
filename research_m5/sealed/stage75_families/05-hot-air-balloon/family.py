"""Hot air balloon altitude.

A balloon starts on the ground and the pilot's actions are read one by one.
"burn" fires the burner and lifts the balloon one level, but it cannot rise
above the highest level; "vent" opens the top vent and lets it sink one level,
but it cannot sink below the ground; "wait" changes nothing. The four levels,
from lowest to highest, are "ground", "low", "mid" and "high". The answer is
the name of the level the balloon is at after the whole sequence.
"""

ALPHABET = ["burn", "vent", "wait"]

_LEVELS = ["ground", "low", "mid", "high"]


def answer(seq):
    level = 0
    for tok in seq:
        if tok == "burn":
            if level < 3:
                level += 1
        elif tok == "vent":
            if level > 0:
                level -= 1
    return _LEVELS[level]
