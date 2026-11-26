"""School bus route.

A school bus drives its route and starts out empty, in pick-up mode. Each
token is one event on the route. "stop": the bus halts at a house; in pick-up
mode one child gets on, in drop-off mode one child gets off and goes home
(if the bus is empty, nothing happens). "turn": the driver turns the bus
around, which switches between pick-up mode and drop-off mode (the children
aboard stay aboard). "school": the bus passes the school and every child
currently aboard gets off there; the bus is then empty but keeps its mode.
The answer is the total number of children who were let off at school over
the whole route, written as a decimal number (children still aboard at the
end, or let off at home, do not count).
"""

ALPHABET = ["stop", "turn", "school"]


def answer(seq):
    pickup = True      # current mode: pick-up (True) or drop-off (False)
    aboard = 0         # children on the bus right now (no upper limit)
    at_school = 0      # children let off at school so far (no upper limit)
    for tok in seq:
        if tok == "stop":
            if pickup:
                aboard += 1
            elif aboard > 0:
                aboard -= 1
        elif tok == "turn":
            pickup = not pickup
        elif tok == "school":
            at_school += aboard
            aboard = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(at_school)
