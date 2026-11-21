"""Concert hall: who carries the melody when the music stops.

Four orchestra sections sit in a ring on the stage, clockwise in the order
strings, winds, brass, percussion (then back to strings). The melody starts
with the strings and the conductor's baton starts out pointing clockwise.
Read the tokens in order: "pass" hands the melody to the next section in the
direction the baton currently points; "flip" reverses the baton's direction
(the melody stays where it is); "solo" sends the melody straight across the
ring to the opposite section (two seats away, whatever the baton direction);
"tutti" is a full-orchestra chord after which the melody goes back to the
strings, while the baton keeps whatever direction it had. The answer is the
name of the section holding the melody after the last token.
"""

ALPHABET = ["pass", "flip", "solo", "tutti"]

_SECTIONS = ["strings", "winds", "brass", "percussion"]


def answer(seq):
    pos = 0          # index into _SECTIONS; strings
    step = 1         # +1 clockwise, -1 counter-clockwise
    for tok in seq:
        if tok == "pass":
            pos = (pos + step) % 4
        elif tok == "flip":
            step = -step
        elif tok == "solo":
            pos = (pos + 2) % 4
        elif tok == "tutti":
            pos = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _SECTIONS[pos]
