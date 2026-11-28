"""Jewelry workshop bracelet bench.

A jeweler works at one open strand, which starts empty.  Reading the tokens in
order: "bead" threads one more bead onto the open strand; "snip" cuts the open
strand and all its beads fall back into the bowl, so the strand is empty again
(snipping an empty strand changes nothing); "clasp" tries to close the strand:
if it holds at least 3 beads, the clasp is fastened, one bracelet is finished
and a new empty strand is started, but if it holds fewer than 3 beads the clasp
cannot be used and is dropped into the spare-clasp tray (the beads stay on the
strand).  Beads left on the strand at the end are not counted.  The answer is
"B/S", where B is the number of finished bracelets and S is the number of
clasps in the spare tray at the end, e.g. "2/1".
"""

ALPHABET = ["bead", "clasp", "snip"]


def answer(seq):
    strand = 0
    bracelets = 0
    spares = 0
    for tok in seq:
        if tok == "bead":
            strand += 1
        elif tok == "snip":
            strand = 0
        elif tok == "clasp":
            if strand >= 3:
                bracelets += 1
                strand = 0
            else:
                spares += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%d/%d" % (bracelets, spares)
