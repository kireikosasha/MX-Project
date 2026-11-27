"""Salt works with one evaporation pan.

The pan starts dry and empty: brine depth 0, no salt crust, no sacks filled.
Read the tokens in order.  "flood" opens the sluice and fills the pan with
brine to its full depth of 2.  "sun" is a hot day: if the pan holds any brine,
the depth drops by 1 and one new layer of salt crust forms; on a dry pan the
sun does nothing.  "rain" dissolves one layer of crust (if there is any) and
also raises the brine depth by 1, never above 2.  "rake" harvests: only if the
pan is dry (depth 0), every layer of crust currently in the pan is raked into
sacks (one sack per layer) and the pan is left bare; if any brine is left the
rake does nothing.  The answer is the total number of sacks filled by the end
of the sequence, written as a decimal number (crust still lying in the pan at
the end does not count).
"""

ALPHABET = ["flood", "sun", "rain", "rake"]

_MAX_DEPTH = 2


def answer(seq):
    depth = 0
    crust = 0
    sacks = 0
    for tok in seq:
        if tok == "flood":
            depth = _MAX_DEPTH
        elif tok == "sun":
            if depth > 0:
                depth -= 1
                crust += 1
        elif tok == "rain":
            if crust > 0:
                crust -= 1
            if depth < _MAX_DEPTH:
                depth += 1
        elif tok == "rake":
            if depth == 0:
                sacks += crust
                crust = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sacks)
