"""A restaurant's reservation book over one evening.

The host keeps two numbers in mind, both starting at 0: the bookings still
in the book (parties written down and still expected) and the free tables.
'book' writes a new party into the book (bookings +1). 'cancel' crosses one
party out of the book (bookings -1; ignored when the book is empty).
'leave' means a seated party finishes and goes, so a table becomes free
(free tables +1). 'walkin' is a party arriving without a booking: the host
holds back one free table for every booking in the book, so the walk-in is
seated only if free tables are more than bookings, and then it takes a
table (free tables -1); otherwise it is turned away and nothing changes.
The answer compares the two numbers after the last token: 'short' if there
are fewer free tables than bookings, 'exact' if they are equal, and
'spare' if there are more free tables than bookings.
"""

ALPHABET = ["book", "cancel", "leave", "walkin"]


def answer(seq):
    bookings = 0
    free = 0
    for tok in seq:
        if tok == "book":
            bookings += 1
        elif tok == "cancel":
            if bookings > 0:
                bookings -= 1
        elif tok == "leave":
            free += 1
        elif tok == "walkin":
            if free > bookings:
                free -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if free < bookings:
        return "short"
    if free == bookings:
        return "exact"
    return "spare"
