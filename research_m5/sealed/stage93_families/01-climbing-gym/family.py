"""Auto-belay wall at a climbing gym.

A single auto-belay rope hangs on the wall, and climbers wait for it in a
line; at the start the line is empty, nobody is on the wall and nobody has
topped out. Read the tokens in order: "join" means one more climber joins the
back of the line; "leave" means the climber at the back of the line gives up
and goes home (nothing happens if the line is empty); "clip" means the climber
at the front of the line clips in and starts climbing, but only if the rope is
free and someone is waiting (otherwise nothing happens); "top" means the
climber on the wall reaches the top, lowers off and is done for the day, which
counts as one send and frees the rope (nothing happens if nobody is on the
wall); "fall" means the climber on the wall comes off, frees the rope and goes
to the back of the line to try again (nothing happens if nobody is on the
wall). The answer is the total number of sends after the whole sequence,
written as a decimal number such as "0" or "7".
"""

ALPHABET = ["join", "leave", "clip", "top", "fall"]


def answer(seq):
    line = 0          # climbers waiting for the rope
    on_wall = False   # is someone clipped in right now
    sends = 0         # climbers who reached the top
    for tok in seq:
        if tok == "join":
            line += 1
        elif tok == "leave":
            if line > 0:
                line -= 1
        elif tok == "clip":
            if not on_wall and line > 0:
                line -= 1
                on_wall = True
        elif tok == "top":
            if on_wall:
                on_wall = False
                sends += 1
        elif tok == "fall":
            if on_wall:
                on_wall = False
                line += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sends)
