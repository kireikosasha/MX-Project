"""Data center job rotation.

A rack of three servers, A, B and C, takes jobs from one shared queue
in a fixed rotation A, B, C, A, ...; the queue starts empty, the room
starts cool (heat 0) and server A is first in line. Read the tokens in
order: "job" adds one job to the end of the queue; "hot" raises the
room heat by one level, but it never goes above 2; "fan" brings the
heat back to 0; "tick" is one work cycle: if the heat is below 2 and
the queue holds at least one job, the server whose turn it is finishes
one job (the queue shrinks by one) and the turn passes to the next
server in the rotation, otherwise (the room is overheated at heat 2,
or the queue is empty) nothing happens. The answer is the name of the
server whose turn it is after the last token: "A", "B" or "C".
"""

ALPHABET = ["job", "tick", "hot", "fan"]

_SERVERS = ["A", "B", "C"]


def answer(seq):
    queue = 0
    heat = 0
    turn = 0
    for tok in seq:
        if tok == "job":
            queue += 1
        elif tok == "hot":
            if heat < 2:
                heat += 1
        elif tok == "fan":
            heat = 0
        elif tok == "tick":
            if heat < 2 and queue > 0:
                queue -= 1
                turn = (turn + 1) % 3
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _SERVERS[turn]
