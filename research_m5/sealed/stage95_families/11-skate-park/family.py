"""Mini-ramp line at a skate park.

The park opens with nobody in line at the top of the mini-ramp and nobody
riding it; read the tokens in order. "arrive": a new skater joins the back of
the line. "drop": if nobody is riding and the line is not empty, the skater at
the front of the line drops in and is now the rider (the line gets one
shorter); if someone is already riding, or the line is empty, nothing happens
(only one rider at a time). "land": if someone is riding, they finish the run
cleanly, roll out and rejoin the back of the line (the ramp is free again);
otherwise nothing happens. "bail": if someone is riding, they slam, pick up
their board and go home (the ramp is free again, and they do not rejoin the
line); otherwise nothing happens. The answer is the number of skaters waiting
in line when the sequence ends (the rider, if any, is not counted), written as
a decimal string such as "0" or "7".
"""

ALPHABET = ["arrive", "drop", "land", "bail"]


def answer(seq):
    line = 0
    riding = False
    for tok in seq:
        if tok == "arrive":
            line += 1
        elif tok == "drop":
            if not riding and line > 0:
                line -= 1
                riding = True
        elif tok == "land":
            if riding:
                riding = False
                line += 1
        elif tok == "bail":
            if riding:
                riding = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(line)
