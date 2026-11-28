"""Recording studio: takes left on the reel.

A singer works in a booth while the engineer runs the tape machine, whose
record button starts switched off. Each token is one event of the session:
"arm" flips the record button (off becomes on, on becomes off); "take" is the
singer performing once, which adds one good take to the reel only if the
record button is on at that moment (otherwise it is just a rehearsal and
nothing is kept); "flub" is the singer botching a performance, and if the
record button is on the engineer erases the most recent take on the reel
(nothing happens if the reel is empty or the button is off). The answer is
the number of takes on the reel after the last token, written as a decimal
string such as "0" or "3".
"""

ALPHABET = ["arm", "take", "flub"]


def answer(seq):
    armed = False
    takes = 0
    for tok in seq:
        if tok == "arm":
            armed = not armed
        elif tok == "take":
            if armed:
                takes += 1
        elif tok == "flub":
            if armed and takes > 0:
                takes -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(takes)
