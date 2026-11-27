"""Knitting circle: whose scarf is longest when the circle breaks up?

Three knitters, ann, bea and cat, sit in a circle and every scarf starts with
no rows. Reading the tokens in order: a name token means that knitter finishes
one more row on her scarf, and she becomes the one who knitted most recently.
The token "frog" means the knitter who knitted most recently (the last name
seen so far) rips out one row of her own scarf; a scarf with no rows cannot
lose any, and a "frog" before anyone has knitted does nothing. A "frog" does
not change who knitted most recently, so several frogs in a row all come off
the same scarf. At the end the answer is the name of the knitter whose scarf
has the most rows; if two or three knitters share the largest number of rows
(and it is above zero) the answer is "tie", and if every scarf has no rows the
answer is "none".
"""

ALPHABET = ["ann", "bea", "cat", "frog"]

_KNITTERS = ("ann", "bea", "cat")


def answer(seq):
    rows = {name: 0 for name in _KNITTERS}
    last = None
    for tok in seq:
        if tok == "frog":
            if last is not None and rows[last] > 0:
                rows[last] -= 1
        elif tok in rows:
            rows[tok] += 1
            last = tok
        else:
            raise ValueError("unknown token: %r" % (tok,))
    best = max(rows.values())
    if best == 0:
        return "none"
    leaders = [name for name in _KNITTERS if rows[name] == best]
    if len(leaders) > 1:
        return "tie"
    return leaders[0]
