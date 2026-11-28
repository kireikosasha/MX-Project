"""Carpentry workshop: stools built by the end of the day.

The workshop starts with an empty pile of boards, no cut legs, and a freshly
sharpened saw. Read the tokens in order. "board": a rough board is delivered
and placed on the pile. "saw": if the pile holds at least one board and the
saw has made fewer than 3 cuts since it was last sharpened, one board is taken
from the pile and cut into 2 legs (the saw's cut count goes up by 1);
otherwise (empty pile or blunt saw) nothing happens. "file": the saw is
sharpened, so its cut count goes back to 0 (boards and legs are untouched).
"glue": if at least 3 legs are waiting, 3 of them are glued into one stool;
otherwise nothing happens. The answer is the number of stools finished after
the last token, written as a decimal number (e.g. "0", "4").
"""

ALPHABET = ["board", "saw", "file", "glue"]

_MAX_CUTS = 3
_LEGS_PER_BOARD = 2
_LEGS_PER_STOOL = 3


def answer(seq):
    pile = 0
    cuts = 0
    legs = 0
    stools = 0
    for tok in seq:
        if tok == "board":
            pile += 1
        elif tok == "saw":
            if pile > 0 and cuts < _MAX_CUTS:
                pile -= 1
                cuts += 1
                legs += _LEGS_PER_BOARD
        elif tok == "file":
            cuts = 0
        elif tok == "glue":
            if legs >= _LEGS_PER_STOOL:
                legs -= _LEGS_PER_STOOL
                stools += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(stools)
