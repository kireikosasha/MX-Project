"""Vineyard harvest and cask rotation.

A single row of vines starts the season with ripeness 0 (ripeness runs from 0 to 3),
and the cellar has three casks used in a fixed rotation: oak, then chestnut, then
acacia, then back to oak; the first ripe crate goes into oak. Read the tokens in
order: "sun" raises ripeness by 1 (it never goes above 3); "rain" lowers it by 1
(it never goes below 0); "frost" drops it straight to 0; "pick" harvests the row:
if ripeness is 2 or more at that moment the crate is good and is poured into the
current cask, after which the rotation moves on to the next cask, while a pick at
ripeness 0 or 1 gives sour grapes that are thrown away and do not move the
rotation; either way, after a pick the ripeness goes back to 0. The answer is the
name of the cask that will receive the next good crate once the whole sequence
has been read ("oak", "chestnut" or "acacia").
"""

ALPHABET = ["sun", "rain", "frost", "pick"]

_CASKS = ["oak", "chestnut", "acacia"]


def answer(seq):
    ripeness = 0
    cask = 0
    for tok in seq:
        if tok == "sun":
            if ripeness < 3:
                ripeness += 1
        elif tok == "rain":
            if ripeness > 0:
                ripeness -= 1
        elif tok == "frost":
            ripeness = 0
        elif tok == "pick":
            if ripeness >= 2:
                cask = (cask + 1) % 3
            ripeness = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _CASKS[cask]
