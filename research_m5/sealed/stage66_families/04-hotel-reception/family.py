"""Hotel reception: a small guesthouse with exactly three rooms, all free when the
sequence starts. The receptionist reads the events in order. "arrive" means a guest
walks in: if at least one room is free the guest takes it, otherwise the guest is
turned away and nothing changes. "depart" means a guest checks out: if any room is
occupied one room becomes free, otherwise (no one is staying) nothing changes.
"inquire" is someone asking a question at the desk and changes nothing. The answer
is how many rooms are occupied after the last event, given as "empty" (0), "one"
(1), "two" (2) or "full" (3).
"""

ALPHABET = ["arrive", "depart", "inquire"]

_ROOMS = 3
_NAMES = ["empty", "one", "two", "full"]


def answer(seq):
    occupied = 0
    for tok in seq:
        if tok == "arrive":
            if occupied < _ROOMS:
                occupied += 1
        elif tok == "depart":
            if occupied > 0:
                occupied -= 1
        elif tok == "inquire":
            pass
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _NAMES[occupied]
