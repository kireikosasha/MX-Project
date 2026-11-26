"""A bank of four parcel lockers, read one event at a time.

The bank has 4 lockers, starts empty and working, and nobody has been turned
away yet. "drop": a courier brings a parcel; if the bank is working and at
least one of the 4 lockers is free, the parcel goes into a locker, otherwise
(bank jammed, or all 4 lockers full) the courier is turned away and takes the
parcel back to the depot. "pick": a customer comes to collect; if the bank is
working and at least one locker holds a parcel, the customer empties that
locker, otherwise (bank jammed, or all lockers empty) the customer is turned
away. "jam": the bank's screen freezes and the bank becomes jammed (no effect
if already jammed); while jammed the lockers keep their contents. "fix": a
technician gets the bank working again (no effect if already working). After
the whole sequence, compare how many couriers and how many customers were
turned away in total: the answer is "couriers" if more couriers were turned
away, "customers" if more customers were turned away, and "even" if the two
totals are equal.
"""

ALPHABET = ["drop", "pick", "jam", "fix"]

LOCKERS = 4


def answer(seq):
    full = 0               # lockers currently holding a parcel (0..4)
    jammed = False
    couriers_away = 0      # unbounded
    customers_away = 0     # unbounded
    for tok in seq:
        if tok == "drop":
            if not jammed and full < LOCKERS:
                full += 1
            else:
                couriers_away += 1
        elif tok == "pick":
            if not jammed and full > 0:
                full -= 1
            else:
                customers_away += 1
        elif tok == "jam":
            jammed = True
        elif tok == "fix":
            jammed = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if couriers_away > customers_away:
        return "couriers"
    if customers_away > couriers_away:
        return "customers"
    return "even"
