"""Candle factory dipping line.

A worker reads a log of factory events. At the start the wax pot is cold and
there are no wicks, no candles and no boxes. "wick" hangs one new bare wick
on the rack. "heat" makes the wax pot hot (it stays hot if it already was).
"dip" plunges the whole rack into the pot: if the pot is hot, every bare wick
currently hanging becomes a finished candle (waiting to be packed) and the
rack is left with no bare wicks; whether or not anything was dipped, the pot
is cold afterwards. If the pot was already cold, "dip" changes nothing.
"pack" puts one finished waiting candle into a box, if there is at least one
waiting; otherwise it does nothing. The answer is the total number of
candles packed into boxes by the end of the log, written as a decimal number
(e.g. "0", "7").
"""

ALPHABET = ["wick", "heat", "dip", "pack"]


def answer(seq):
    hot = False
    bare = 0
    ready = 0
    packed = 0
    for tok in seq:
        if tok == "wick":
            bare += 1
        elif tok == "heat":
            hot = True
        elif tok == "dip":
            if hot:
                ready += bare
                bare = 0
            hot = False
        elif tok == "pack":
            if ready > 0:
                ready -= 1
                packed += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(packed)
