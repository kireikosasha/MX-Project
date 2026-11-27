"""A small tannery moves hides through three stages: raw, limed and tanned.
Read the tokens in order. "hide" brings one new raw hide into the yard.
"lime" takes one raw hide (if there is any) and puts it in the lime pit,
making it limed; with no raw hide it does nothing. "tan" takes one limed hide
(if any) and puts it in the tan vat, making it tanned; otherwise nothing.
"sell" sends one tanned hide (if any) out of the yard for good; otherwise
nothing. After the last token, the answer is the stage that holds the most
hides ("raw", "limed" or "tanned"), with a tie going to the earlier stage
(raw before limed before tanned), or "empty" if no hide is left in the yard.
"""

ALPHABET = ["hide", "lime", "tan", "sell"]


def answer(seq):
    raw = limed = tanned = 0
    for tok in seq:
        if tok == "hide":
            raw += 1
        elif tok == "lime":
            if raw > 0:
                raw -= 1
                limed += 1
        elif tok == "tan":
            if limed > 0:
                limed -= 1
                tanned += 1
        elif tok == "sell":
            if tanned > 0:
                tanned -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if raw == limed == tanned == 0:
        return "empty"
    best = max(raw, limed, tanned)
    if raw == best:
        return "raw"
    if limed == best:
        return "limed"
    return "tanned"
