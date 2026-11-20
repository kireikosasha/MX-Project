"""Library shelf with a waiting list.

A small library starts empty: no books on the shelf, none on loan, and nobody
waiting. Read the tokens in order. "donate" brings in a new book: if a reader
is waiting it goes straight to the first waiting reader (now on loan),
otherwise it is put on the shelf. "borrow" is a reader asking for a book: if
the shelf has one it goes out on loan, otherwise the reader joins the waiting
list. "return" brings back one book on loan (ignored if no book is on loan):
if a reader is waiting it is handed straight to them (still on loan),
otherwise it goes back on the shelf. "weed" discards one worn book from the
shelf (ignored if the shelf is empty). The answer is the number of books
sitting on the shelf after the whole sequence, written as a decimal string.
"""

ALPHABET = ["donate", "borrow", "return", "weed"]


def answer(seq):
    shelf = 0
    out = 0
    wait = 0
    for tok in seq:
        if tok == "donate":
            if wait > 0:
                wait -= 1
                out += 1
            else:
                shelf += 1
        elif tok == "borrow":
            if shelf > 0:
                shelf -= 1
                out += 1
            else:
                wait += 1
        elif tok == "return":
            if out > 0:
                out -= 1
                if wait > 0:
                    wait -= 1
                    out += 1
                else:
                    shelf += 1
        elif tok == "weed":
            if shelf > 0:
                shelf -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(shelf)
