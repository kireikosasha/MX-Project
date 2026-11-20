"""Tailor's seam.

A tailor works along a seam, starting with an unthreaded needle and the
cloth face up. "thread" threads the needle and "snip" cuts the thread so the
needle is unthreaded again (each has no effect if the needle is already in
that state). "press" turns the cloth over, flipping between face up and face
down. "stitch" does nothing while the needle is unthreaded; with a threaded
needle it adds one stitch to the seam when the cloth is face up and takes one
stitch out (a backstitch) when the cloth is face down. The seam's tension is
the running stitch total counted modulo 3 (so taking a stitch out of a total
of 0 gives 2): the answer is "flat" for 0, "puckered" for 1 and "gathered"
for 2, read after the whole sequence.
"""

ALPHABET = ["thread", "stitch", "snip", "press"]

_NAMES = ("flat", "puckered", "gathered")


def answer(seq):
    threaded = False
    face_up = True
    total = 0
    for tok in seq:
        if tok == "thread":
            threaded = True
        elif tok == "snip":
            threaded = False
        elif tok == "press":
            face_up = not face_up
        elif tok == "stitch":
            if threaded:
                total = (total + 1) % 3 if face_up else (total - 1) % 3
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _NAMES[total]
