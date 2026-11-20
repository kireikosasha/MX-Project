"""Bus route: stranded passengers.

A small bus with room for exactly 2 passengers starts its run empty and reads
the sequence as events along the route. "board" means a person at the stop
wants to get on: if fewer than 2 passengers are aboard they get on, otherwise
the bus is full and that person is left stranded at the stop. "alight" means
one passenger gets off (nothing happens if the bus is already empty).
"terminus" means the bus reaches the end of the line and everyone aboard gets
off, leaving it empty. "wait" means the bus pauses at a stop where nothing
happens. The answer is the total number of people left stranded over the whole
run, written as a decimal string (for example "0", "2", "17").
"""

ALPHABET = ["board", "alight", "terminus", "wait"]

CAPACITY = 2


def answer(seq):
    load = 0
    stranded = 0
    for tok in seq:
        if tok == "board":
            if load < CAPACITY:
                load += 1
            else:
                stranded += 1
        elif tok == "alight":
            if load > 0:
                load -= 1
        elif tok == "terminus":
            load = 0
        elif tok == "wait":
            pass
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(stranded)
