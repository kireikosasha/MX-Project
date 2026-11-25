"""Taxi rank.

A small taxi rank outside a station is watched one event at a time. "cab" means
a taxi pulls in: if passengers are queueing, it takes the first one and leaves
at once; otherwise it parks on the rank, unless the rank already holds 4 taxis,
in which case it drives on. "fare" means a passenger walks up: if a taxi is
parked, the passenger takes it and both leave; otherwise the passenger joins
the queue, unless 5 people are already queueing, in which case the newcomer
gives up and walks away. "bus" means a bus stops by the rank: every queueing
passenger boards it (the queue empties), while parked taxis stay where they
are. The rank starts with no taxis and no queue. The answer describes the rank
after the last event: "N cabs" if N taxis are parked, "N waiting" if N
passengers are queueing, or "empty" if there is neither.
"""

ALPHABET = ["cab", "fare", "bus"]

MAX_CABS = 4
MAX_QUEUE = 5


def answer(seq):
    cabs = 0
    queue = 0
    for tok in seq:
        if tok == "cab":
            if queue > 0:
                queue -= 1
            elif cabs < MAX_CABS:
                cabs += 1
        elif tok == "fare":
            if cabs > 0:
                cabs -= 1
            elif queue < MAX_QUEUE:
                queue += 1
        elif tok == "bus":
            queue = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if cabs > 0:
        return "%d cabs" % cabs
    if queue > 0:
        return "%d waiting" % queue
    return "empty"
