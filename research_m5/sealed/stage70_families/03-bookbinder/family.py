"""Bookbinder's sewing bench.

A bookbinder builds a book block one signature at a time. There is always one
open signature on the bench, which starts empty. "sheet" lays one more folded
sheet into the open signature. "spoil" takes one sheet out of the open
signature because it was misprinted (nothing happens if the signature is
empty). "sew" sews the open signature onto the book block, but only if it holds
at least two sheets; it then counts as one sewn signature and a new, empty
signature is opened. A "sew" on a signature with fewer than two sheets does
nothing (the sheet stays on the bench). The answer is the number of signatures
sewn onto the block by the end, counted modulo 4, written as a digit
"0", "1", "2" or "3".
"""

ALPHABET = ["sheet", "spoil", "sew"]


def answer(seq):
    open_sheets = 0
    sewn = 0
    for tok in seq:
        if tok == "sheet":
            open_sheets += 1
        elif tok == "spoil":
            if open_sheets > 0:
                open_sheets -= 1
        elif tok == "sew":
            if open_sheets >= 2:
                sewn += 1
                open_sheets = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sewn % 4)
