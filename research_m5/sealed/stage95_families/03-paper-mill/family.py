"""Paper mill line.

The mill keeps a vat of pulp batches and a felt holding wet sheets, both
starting empty, and a tally of finished dry sheets starting at zero.  Read
the tokens in order: "pulp" pours one batch into the vat; "couch" lifts one
batch from the vat and lays it on the felt as a wet sheet (nothing happens
if the vat is empty); "dry" sends one wet sheet from the felt through the
dryer, adding one to the finished tally (nothing happens if the felt is
empty); "tear" means one wet sheet on the felt rips and is thrown back into
the vat as one batch of pulp (nothing happens if the felt is empty).  The
vat and the felt have no size limit.  The answer is the number of finished
dry sheets after the last token, written as a decimal number.
"""

ALPHABET = ["pulp", "couch", "dry", "tear"]


def answer(seq):
    vat = 0
    felt = 0
    finished = 0
    for tok in seq:
        if tok == "pulp":
            vat += 1
        elif tok == "couch":
            if vat > 0:
                vat -= 1
                felt += 1
        elif tok == "dry":
            if felt > 0:
                felt -= 1
                finished += 1
        elif tok == "tear":
            if felt > 0:
                felt -= 1
                vat += 1
    return str(finished)
