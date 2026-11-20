"""Parking-garage family.

A small garage has 4 spaces and starts empty; the tokens are events read in
order. "car": a car arrives and parks in 1 space if at least 1 space is free,
otherwise it is turned away. "van": a van arrives and parks in 2 spaces if at
least 2 spaces are free, otherwise it is turned away. "leave": the vehicle that
has been parked the longest (the earliest arrival still inside) drives out and
frees its spaces; nothing happens if the garage is empty. "tow": every van
currently parked is towed away, freeing its spaces, while the cars stay where
they are and keep their arrival order. The answer is the number of free spaces
left after the last event, written as a digit "0" to "4".
"""

from collections import deque

ALPHABET = ["car", "van", "leave", "tow"]

CAPACITY = 4
SIZE = {"car": 1, "van": 2}


def answer(seq):
    parked = deque()  # vehicles in arrival order, oldest on the left
    used = 0
    for tok in seq:
        if tok in SIZE:
            if CAPACITY - used >= SIZE[tok]:
                parked.append(tok)
                used += SIZE[tok]
        elif tok == "leave":
            if parked:
                used -= SIZE[parked.popleft()]
        elif tok == "tow":
            parked = deque(v for v in parked if v != "van")
            used = len(parked)
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(CAPACITY - used)
