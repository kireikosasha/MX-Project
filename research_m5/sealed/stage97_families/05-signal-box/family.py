"""Single-line signal box.

A signalman works a single-track section of line that has a waiting loop at
each end. The line starts set for the east end. Reading the tokens in order:
"east" means a train arrives at the east end and waits there; "west" means a
train arrives at the west end and waits there; "lever" means the signalman
pulls the lever that reverses which end the line is set for (east becomes
west, west becomes east); "clear" means the section has cleared, so one train
waiting at the end the line is currently set for is sent through and leaves
(if no train waits at that end, nothing happens, even if trains wait at the
other end). After the whole sequence, the answer is the end with more trains
still waiting: "east", "west", or "even" if both ends hold the same number
(including none).
"""

ALPHABET = ["east", "west", "lever", "clear"]


def answer(seq):
    waiting = {"east": 0, "west": 0}
    line_set = "east"
    for tok in seq:
        if tok == "east" or tok == "west":
            waiting[tok] += 1
        elif tok == "lever":
            line_set = "west" if line_set == "east" else "east"
        elif tok == "clear":
            if waiting[line_set] > 0:
                waiting[line_set] -= 1
    if waiting["east"] > waiting["west"]:
        return "east"
    if waiting["west"] > waiting["east"]:
        return "west"
    return "even"
