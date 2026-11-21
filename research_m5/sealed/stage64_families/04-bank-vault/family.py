"""Bank-vault combination dial.

The vault door has a dial with five positions numbered 0 to 4; it starts at
position 0 and the door starts shut.  Reading the tokens in order: "left"
turns the dial one step down and "right" one step up (both wrap around, so
left from 0 gives 4 and right from 4 gives 0).  "pull" yanks the handle: if
the dial is at position 3 at that moment the bolt gives and the door flips
state (a shut door swings open, an open door is pushed shut again); at any
other position the pull does nothing.  "alarm" slams the door shut (it stays
shut if already shut) without moving the dial.  The answer is the state of
the door after the last token: "open" or "shut".
"""

ALPHABET = ["left", "right", "pull", "alarm"]


def answer(seq):
    dial = 0
    is_open = False
    for tok in seq:
        if tok == "left":
            dial = (dial - 1) % 5
        elif tok == "right":
            dial = (dial + 1) % 5
        elif tok == "pull":
            if dial == 3:
                is_open = not is_open
        elif tok == "alarm":
            is_open = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "open" if is_open else "shut"
