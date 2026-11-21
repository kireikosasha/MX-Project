"""Weaving loom: shuttle side and the colour of the next row of cloth.

A hand loom starts with its shuttle on the left and a freshly opened shed.
"treadle" changes the shed (opens a new one).  "throw" sends the shuttle to the
other side; it lays a loose pick only if the shed has been changed since the
last pick was laid (the very first throw counts as a fresh shed), otherwise the
yarn just slides back out and nothing is laid.  "beat" presses all loose picks
into the cloth, each becoming one woven row; loose picks never beaten are not
rows.  Rows use yarn in the repeating order red, gold, blue, so the colour of
the next row is red, gold or blue as the number of woven rows is 0, 1 or 2
modulo 3.  The answer is "<colour of the next row>-<side the shuttle is on>",
for example "gold-right".
"""

ALPHABET = ["throw", "treadle", "beat"]

_COLOURS = ("red", "gold", "blue")


def answer(seq):
    side = 0          # 0 = left, 1 = right
    fresh = True      # shed changed since the last laid pick
    loose = 0         # picks laid but not yet beaten (only mod 3 matters)
    woven = 0         # woven rows, kept modulo 3
    for tok in seq:
        if tok == "treadle":
            fresh = True
        elif tok == "throw":
            side ^= 1
            if fresh:
                loose = (loose + 1) % 3
                fresh = False
        elif tok == "beat":
            woven = (woven + loose) % 3
            loose = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%s-%s" % (_COLOURS[woven], "right" if side else "left")
