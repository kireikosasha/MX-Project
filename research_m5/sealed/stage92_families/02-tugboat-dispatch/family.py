"""Tugboat dispatch office.

The dispatch office opens with 2 idle tugs at the dock, no ships waiting and
clear weather, then reads the log token by token: "ship" means a ship calls
for a tow, and in clear weather it takes an idle tug if there is one (the tug
leaves the dock) and otherwise joins the waiting list; "tug" means a tug
reports free, and in clear weather it goes straight to a waiting ship if any
(that ship leaves the waiting list) and otherwise stays idle at the dock;
"fog" closes the harbour, and while it is foggy no tug moves, so every "ship"
joins the waiting list and every "tug" stays idle at the dock (a second "fog"
changes nothing); "clear" lifts the fog and at once pairs idle tugs with
waiting ships, one with one, as many pairs as possible (in clear weather it
changes nothing). The answer is the state of the dock after the last token,
written as "idle I waiting W", where I is the number of idle tugs at the dock
and W the number of ships on the waiting list.
"""

ALPHABET = ["ship", "tug", "fog", "clear"]


def answer(seq):
    idle = 2
    waiting = 0
    foggy = False
    for tok in seq:
        if tok == "ship":
            if not foggy and idle > 0:
                idle -= 1
            else:
                waiting += 1
        elif tok == "tug":
            if not foggy and waiting > 0:
                waiting -= 1
            else:
                idle += 1
        elif tok == "fog":
            foggy = True
        elif tok == "clear":
            foggy = False
            k = min(idle, waiting)
            idle -= k
            waiting -= k
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "idle %d waiting %d" % (idle, waiting)
