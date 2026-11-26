"""Warehouse loading dock.

Trucks wait in a single line at one loading bay; the truck at the front of the
line is the one at the bay, and every truck can hold 3 crates. Read the tokens
in order. "truck": a new truck joins the back of the line; if the bay was
empty it pulls straight in. "crate": a crate arrives; if a truck is at the bay
the crate goes onto that truck, otherwise it is set down on the dock floor.
"go": the truck at the bay leaves as it is, however much it holds (nothing
happens if the bay is empty). Whenever a truck pulls into the bay it first
takes crates from the floor, one at a time, until it is full or the floor is
empty, and whenever a truck at the bay reaches 3 crates it leaves at once.
After any truck leaves, the next truck in line (if any) pulls into the bay.
The answer is the number of trucks that left the dock full (with exactly 3
crates), written as a decimal number, e.g. "0", "2", "17".
"""

ALPHABET = ["truck", "crate", "go"]

CAPACITY = 3


def answer(seq):
    line = 0   # trucks at the dock; the first of them is at the bay
    floor = 0  # crates waiting on the dock floor
    load = 0   # crates on the truck at the bay
    full = 0   # trucks that have left full

    def pull_in():
        # The truck now at the front of the line (if any) enters the bay,
        # takes crates from the floor, and leaves at once if that fills it.
        nonlocal line, floor, load, full
        while line > 0:
            take = min(CAPACITY, floor)
            floor -= take
            load = take
            if load < CAPACITY:
                return
            full += 1
            line -= 1
            load = 0

    for tok in seq:
        if tok == "truck":
            line += 1
            if line == 1:
                pull_in()
        elif tok == "crate":
            if line > 0:
                load += 1
                if load == CAPACITY:
                    full += 1
                    line -= 1
                    load = 0
                    pull_in()
            else:
                floor += 1
        elif tok == "go":
            if line > 0:
                line -= 1
                load = 0
                pull_in()
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(full)
