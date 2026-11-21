"""Flower-shop bucket.

A florist keeps one bucket of cut flowers outside the shop; it starts empty with its
water level at 1 (levels go 0, 1, 2). Reading the day's events in order: "rose" or
"tulip" puts stems of that kind in the bucket; "water" tops the bucket up by one level
(never above 2); "sun" evaporates one level, but if the level is already 0 the stems
dry out and every flower in the bucket is thrown away (the level stays 0); "buyer" is a
customer who takes all the stems of whichever kind still in the bucket was put in most
recently (nothing happens if the bucket is empty). The answer is what is left in the
bucket at closing time: "empty", "roses", "tulips" or "both".
"""

ALPHABET = ["rose", "tulip", "water", "sun", "buyer"]


def answer(seq):
    water = 1
    present = []  # kinds in the bucket, least recently added first
    for tok in seq:
        if tok == "rose" or tok == "tulip":
            if tok in present:
                present.remove(tok)
            present.append(tok)
        elif tok == "water":
            if water < 2:
                water += 1
        elif tok == "sun":
            if water == 0:
                present = []
            else:
                water -= 1
        elif tok == "buyer":
            if present:
                present.pop()
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if not present:
        return "empty"
    if len(present) == 2:
        return "both"
    return "roses" if present[0] == "rose" else "tulips"
