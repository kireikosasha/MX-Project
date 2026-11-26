"""Bicycle courier depot: how many ordinary boxes are still on the shelf at closing.

The depot opens with an empty shelf and no riders waiting. Read the tokens in
order. "rush" (an express envelope) and "box" (an ordinary parcel) are handed in
at the counter: if a rider is waiting at the depot, one waiting rider takes it
straight away; otherwise it is put on the shelf. "rider" means a courier comes
back to the depot looking for work: they take a rush envelope from the shelf if
there is one, otherwise an ordinary box if there is one, and if the shelf is
empty they wait at the depot (riders can pile up without limit, and so can
envelopes and boxes on the shelf). The answer is the number of ordinary boxes
left on the shelf after the last token, written as a decimal string (e.g. "0",
"3").
"""

ALPHABET = ["rush", "box", "rider"]


def answer(seq):
    rush = 0    # rush envelopes on the shelf
    boxes = 0   # ordinary boxes on the shelf
    idle = 0    # riders waiting at the depot
    for tok in seq:
        if tok == "rush":
            if idle:
                idle -= 1
            else:
                rush += 1
        elif tok == "box":
            if idle:
                idle -= 1
            else:
                boxes += 1
        elif tok == "rider":
            if rush:
                rush -= 1
            elif boxes:
                boxes -= 1
            else:
                idle += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(boxes)
