"""Archive holdings after a day of events.

An archive starts empty. Reading the tokens in order: "file" adds one new
document to the shelves; "lend" takes one document off the shelves and lends it
to a reader (nothing happens if the shelves are empty); "back" returns one lent
document to the shelves (nothing happens if nothing is on loan); "flood"
destroys every document currently on the shelves, while documents out on loan
are safe. The answer is the number of documents the archive still owns at the
end (those on the shelves plus those on loan), written as a decimal string.
"""

ALPHABET = ["file", "lend", "back", "flood"]


def answer(seq):
    shelf = 0
    out = 0
    for tok in seq:
        if tok == "file":
            shelf += 1
        elif tok == "lend":
            if shelf > 0:
                shelf -= 1
                out += 1
        elif tok == "back":
            if out > 0:
                out -= 1
                shelf += 1
        elif tok == "flood":
            shelf = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(shelf + out)
