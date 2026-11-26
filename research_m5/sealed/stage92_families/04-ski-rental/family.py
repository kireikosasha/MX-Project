"""Ski rental shop: how many customers are still waiting at closing.

The shop owns exactly 3 pairs of skis, and at opening all 3 are on the rack;
nobody is waiting and the wax bench is empty. Read the tokens in order. "guest":
a customer arrives; if a pair is on the rack they take it and go out on the
slopes, otherwise they join the end of the waiting line. "return": a renter
brings a pair back; this only happens if at least one pair is currently out on
the slopes (otherwise the token is ignored), and the returned pair goes onto
the wax bench, not onto the rack. "wax": the technician finishes one pair from
the wax bench (ignored if the bench is empty); the finished pair is handed to
the first customer in the waiting line, who leaves the line and goes out with
it, or, if nobody is waiting, it goes back on the rack. The answer is the
number of customers still in the waiting line after the last token, written as
a decimal string (for example "0" or "4").
"""

ALPHABET = ["guest", "return", "wax"]

_FLEET = 3


def answer(seq):
    rack = _FLEET
    bench = 0
    out = 0
    waiting = 0
    for tok in seq:
        if tok == "guest":
            if rack > 0:
                rack -= 1
                out += 1
            else:
                waiting += 1
        elif tok == "return":
            if out > 0:
                out -= 1
                bench += 1
        elif tok == "wax":
            if bench > 0:
                bench -= 1
                if waiting > 0:
                    waiting -= 1
                    out += 1
                else:
                    rack += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(waiting)
