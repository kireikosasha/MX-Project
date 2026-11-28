"""Bobsled track timing hut.

The hut keeps three things in mind while reading the day's log token by token.
The ice level starts at 1 and stays between 0 and 3: "frost" raises it by one
(no higher than 3) and "sun" lowers it by one (no lower than 0). "launch" sends
one more sled onto the track, so the number of sleds on the track grows by one.
"arrive" means a sled reaches the bottom, but only if at least one sled is on
the track; otherwise the token is ignored. When a sled does arrive, the number
on the track drops by one and the run is timed: it is a fast run if the ice
level at that moment is 2 or more, and a slow run otherwise. The answer is
"fast" if there were more fast runs than slow runs over the whole log, "slow"
if there were more slow runs, and "even" if the two counts are equal
(including when no sled ever arrived).
"""

ALPHABET = ["launch", "arrive", "frost", "sun"]


def answer(seq):
    ice = 1
    on_track = 0
    balance = 0  # fast runs minus slow runs
    for tok in seq:
        if tok == "frost":
            if ice < 3:
                ice += 1
        elif tok == "sun":
            if ice > 0:
                ice -= 1
        elif tok == "launch":
            on_track += 1
        elif tok == "arrive":
            if on_track > 0:
                on_track -= 1
                if ice >= 2:
                    balance += 1
                else:
                    balance -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "fast"
    if balance < 0:
        return "slow"
    return "even"
