"""Ambulance depot dispatch board.

The depot opens with 2 ambulances parked, none out on a job and no calls
waiting. Read the tokens in order. "call": an emergency call comes in; if an
ambulance is parked it leaves to answer it (one fewer parked, one more out),
otherwise the call joins the waiting list. "back": an ambulance out on a job
returns; if no ambulance is out the token is ignored; if calls are waiting the
returning ambulance immediately leaves again for one of them (one fewer
waiting, the number out stays the same), otherwise it parks (one fewer out,
one more parked). "new": a brand-new ambulance joins the fleet; if calls are
waiting it immediately leaves for one of them (one fewer waiting, one more
out), otherwise it parks (one more parked). The answer, after the whole
sequence, is "wait N" when N > 0 calls are still waiting, otherwise "park N"
where N is the number of ambulances parked at the depot (possibly 0).
"""

ALPHABET = ["call", "back", "new"]


def answer(seq):
    parked, out, waiting = 2, 0, 0
    for tok in seq:
        if tok == "call":
            if parked > 0:
                parked -= 1
                out += 1
            else:
                waiting += 1
        elif tok == "back":
            if out == 0:
                continue
            if waiting > 0:
                waiting -= 1
            else:
                out -= 1
                parked += 1
        elif tok == "new":
            if waiting > 0:
                waiting -= 1
                out += 1
            else:
                parked += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if waiting > 0:
        return "wait %d" % waiting
    return "park %d" % parked
