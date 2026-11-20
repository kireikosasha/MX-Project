"""Single-platform station with small trains.

The tokens are read in order as events at one platform. "train" means a
train pulls in, but only if the platform track is free (if a train is
already standing there the newcomer is turned away and nothing changes);
the new train arrives empty, has room for exactly 2 riders, and at once
takes up to 2 riders from the people waiting on the platform. "rider" means
one person walks onto the platform: if a train is standing there with a free
seat they board it, otherwise they join the people waiting on the platform.
"whistle" means the standing train departs with whoever is aboard, freeing
the track; with no train present it does nothing. The answer is the number
of people still waiting on the platform after the last event, written as a
decimal string (riders sitting in a train that has not yet left do not
count as waiting).
"""

ALPHABET = ["train", "rider", "whistle"]

CAPACITY = 2


def answer(seq):
    waiting = 0
    train_here = False
    seats_free = 0
    for tok in seq:
        if tok == "train":
            if not train_here:
                train_here = True
                take = min(CAPACITY, waiting)
                waiting -= take
                seats_free = CAPACITY - take
        elif tok == "rider":
            if train_here and seats_free > 0:
                seats_free -= 1
            else:
                waiting += 1
        elif tok == "whistle":
            if train_here:
                train_here = False
                seats_free = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(waiting)
