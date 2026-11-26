"""Lazy-river tube dock at a water park.

The river starts open, with nobody waiting in line and no tubes tied up at
the dock. Each `guest` joins the end of the waiting line. Each `tube` that
arrives while the river is open floats off at once carrying one waiting guest
(if the line is empty it floats off empty). `storm` closes the river (no
effect if it is already closed); while the river is closed every arriving
`tube` is tied up at the dock instead of leaving, and guests keep joining the
line. `sun` reopens the river (no effect if it is already open); at that
moment all tied-up tubes float off together, each taking one waiting guest if
any are left, and the dock is empty again. The answer is the number of guests
still waiting in line after the last token, written as a decimal string (for
example "0" or "3"); tubes still tied up when the sequence ends carry nobody.
"""

ALPHABET = ["guest", "tube", "storm", "sun"]


def answer(seq):
    is_open = True
    line = 0
    tied = 0
    for tok in seq:
        if tok == "guest":
            line += 1
        elif tok == "tube":
            if is_open:
                if line > 0:
                    line -= 1
            else:
                tied += 1
        elif tok == "storm":
            is_open = False
        elif tok == "sun":
            if not is_open:
                line = max(0, line - tied)
                tied = 0
                is_open = True
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(line)
