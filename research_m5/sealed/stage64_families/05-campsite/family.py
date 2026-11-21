"""Campfire keeper.

A campfire ring starts empty and unlit, and each token is one event at the
campsite. "log" puts one log in the ring, but the ring holds at most 3 logs, so a
log added to a full ring is set aside and does not count. "match" lights the fire
if it is not already burning and there is at least one log in the ring; otherwise
the match does nothing. "rain" puts the fire out (the logs stay in the ring).
"night" passes one night: if the fire is burning it burns up one log, and if that
leaves the ring empty the fire goes out; an unlit fire burns nothing. The answer
is the state of the fire after the last event, written as "lit-N" or "out-N",
where N (0 to 3) is the number of logs left in the ring.
"""

ALPHABET = ["log", "match", "rain", "night"]

MAX_LOGS = 3


def answer(seq):
    logs = 0
    burning = False
    for tok in seq:
        if tok == "log":
            if logs < MAX_LOGS:
                logs += 1
        elif tok == "match":
            if not burning and logs > 0:
                burning = True
        elif tok == "rain":
            burning = False
        elif tok == "night":
            if burning:
                logs -= 1
                if logs == 0:
                    burning = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return ("lit" if burning else "out") + "-" + str(logs)
