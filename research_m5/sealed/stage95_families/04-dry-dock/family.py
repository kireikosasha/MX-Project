"""Dry dock with one berth.

The dock starts dry and empty, with nobody waiting at the gate. "ship" means a
ship arrives and joins the line at the gate. "flood" fills the dock with water
and "pump" drains it dry; the water stays as it is until the next flood or
pump. "work" repairs the ship in the berth, but only if the dock is dry and a
ship is in it (otherwise it does nothing; repairing an already repaired ship
changes nothing). After every token, if the dock is flooded, two things happen
in this order: a repaired ship in the berth floats out and leaves (it counts as
finished), and then, if the berth is empty and someone is waiting at the gate,
the first waiting ship floats into the berth, unrepaired. So a ship that
arrives while the dock is flooded and empty goes straight in, and an
unrepaired ship never leaves. The answer is the number of finished ships, that
is, ships that left the dock repaired, written as a decimal number.
"""

ALPHABET = ["ship", "flood", "pump", "work"]


def answer(seq):
    flooded = False
    waiting = 0
    berth = None  # None: empty, False: unrepaired ship, True: repaired ship
    finished = 0
    for tok in seq:
        if tok == "ship":
            waiting += 1
        elif tok == "flood":
            flooded = True
        elif tok == "pump":
            flooded = False
        elif tok == "work":
            if not flooded and berth is not None:
                berth = True
        else:
            raise ValueError("unknown token: %r" % (tok,))
        if flooded:
            if berth is True:
                berth = None
                finished += 1
            if berth is None and waiting > 0:
                waiting -= 1
                berth = False
    return str(finished)
