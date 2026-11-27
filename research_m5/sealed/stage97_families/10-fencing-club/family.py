"""A practice bout at a fencing club between the fencer on the Left and the
fencer on the Right, read one call at a time. At the start nobody has scored
and the Left fencer holds the right of way. "beat": a beat on the blade, the
right of way passes to the other fencer (no one scores). "lunge": the fencer
who currently holds the right of way scores one touch. "card": the fencer who
currently holds the right of way is given a penalty card, so the opponent
scores one touch (the right of way does not move). "double": both lights go on
at once, the touches are annulled (no one scores) and the referee gives the
right of way back to the Left fencer. After the whole sequence the answer is
"left" if the Left fencer has more touches, "right" if the Right fencer has
more, and "level" if they have the same number of touches.
"""

ALPHABET = ["beat", "lunge", "card", "double"]


def answer(seq):
    holder = 0  # 0 = Left holds the right of way, 1 = Right holds it
    lead = 0    # Left touches minus Right touches
    for tok in seq:
        if tok == "beat":
            holder = 1 - holder
        elif tok == "lunge":
            lead += 1 if holder == 0 else -1
        elif tok == "card":
            lead += -1 if holder == 0 else 1
        elif tok == "double":
            holder = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if lead > 0:
        return "left"
    if lead < 0:
        return "right"
    return "level"
