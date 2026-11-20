"""Hospital ward with three beds and a lockdown switch.

The ward starts empty (0 of 3 beds occupied) and open. Read the tokens in
order: "admit" puts a new patient into a free bed, but only if the ward is
open and not already full (otherwise the patient is turned away and nothing
changes); "discharge" sends one patient home, freeing a bed (nothing happens
if the ward is empty); "alarm" puts the ward into lockdown (already locked
stays locked); "round" is the doctor's round, which lifts any lockdown and,
if all three beds are occupied at that moment, also discharges one patient
so the ward has a free bed again. The answer is the number of occupied beds
after the last token followed by the ward's status, e.g. "2 open" or
"0 locked".
"""

ALPHABET = ["admit", "discharge", "alarm", "round"]

CAPACITY = 3


def answer(seq):
    beds = 0
    locked = False
    for tok in seq:
        if tok == "admit":
            if not locked and beds < CAPACITY:
                beds += 1
        elif tok == "discharge":
            if beds > 0:
                beds -= 1
        elif tok == "alarm":
            locked = True
        elif tok == "round":
            locked = False
            if beds == CAPACITY:
                beds -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%d %s" % (beds, "locked" if locked else "open")
