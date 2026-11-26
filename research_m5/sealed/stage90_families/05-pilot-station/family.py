"""Harbour pilot station: how many ships had to wait at anchor for a pilot.

The station starts with no pilots on hand, no ships at anchor, and clear
weather. Read the tokens in order. "pilot": a pilot comes back to the station
and is free. "ship": a ship arrives and calls for a pilot; if the weather is
clear and a pilot is free (and no ship is already waiting at anchor), the pilot
boards it at once, otherwise the ship drops anchor to wait and counts as
delayed. "fog": fog comes down and no pilot can board while it lasts (pilots and
ships can both pile up). "clear": the fog lifts (clear while already clear, or
fog while already foggy, changes nothing). Whenever the weather is clear,
waiting ships are boarded one per free pilot, straight away, so in clear
weather free pilots and waiting ships never both remain. The answer is the
total number of ships that were delayed (had to anchor on arrival), as a
decimal string; a delayed ship still counts even after it is boarded later.
"""

ALPHABET = ["ship", "pilot", "fog", "clear"]


def answer(seq):
    free = 0        # pilots on hand at the station
    waiting = 0     # ships at anchor waiting for a pilot
    fog = False
    delayed = 0
    for tok in seq:
        if tok == "pilot":
            free += 1
        elif tok == "ship":
            if not fog and free > 0 and waiting == 0:
                free -= 1
            else:
                waiting += 1
                delayed += 1
        elif tok == "fog":
            fog = True
        elif tok == "clear":
            fog = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
        if not fog and free and waiting:
            m = min(free, waiting)
            free -= m
            waiting -= m
    return str(delayed)
