"""Robot warehouse picking station.

Robots carry shelves along a single aisle to a picking station, where the
shelves wait in a line until a picker is done with them. The aisle starts
open and the station starts with no shelves waiting. Read the tokens in
order: "bring" means a robot tries to deliver one more shelf to the station,
which only succeeds while the aisle is open (while it is blocked, the robot
cannot get through and nothing changes); "pick" means the picker finishes
the shelf at the front, so one waiting shelf leaves the station (if no shelf
is waiting, nothing happens), and this works whether the aisle is open or
blocked; "block" means the aisle becomes blocked (it stays blocked if it
already was); "clear" means the aisle becomes open again (it stays open if
it already was). The answer is the number of shelves waiting at the station
after the whole sequence, written as a decimal string such as "0" or "3".
"""

ALPHABET = ["bring", "pick", "block", "clear"]


def answer(seq):
    waiting = 0
    blocked = False
    for tok in seq:
        if tok == "bring":
            if not blocked:
                waiting += 1
        elif tok == "pick":
            if waiting > 0:
                waiting -= 1
        elif tok == "block":
            blocked = True
        elif tok == "clear":
            blocked = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(waiting)
