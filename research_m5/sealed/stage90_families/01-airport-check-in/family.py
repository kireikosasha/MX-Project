"""Airport check-in desk.

The desk starts with an empty queue, nobody standing at the counter and an
empty baggage belt.  Read the tokens in order: "pax" means a passenger joins
the back of the queue; "call" means the agent finishes with whoever is at the
counter and calls the next person from the front of the queue (if the queue is
empty, the counter is left empty; otherwise the queue gets one shorter and the
counter is occupied); "bag" means the passenger at the counter puts one bag on
the belt (ignored if nobody is at the counter); "load" means a handler takes
one bag off the belt to the plane (ignored if the belt is empty).  After the
last token, compare the number of people still waiting in the queue (not
counting anyone at the counter) with the number of bags still on the belt: the
answer is "queue" if more people are waiting, "belt" if more bags are waiting,
and "even" if the two numbers are equal.
"""

ALPHABET = ["pax", "call", "bag", "load"]


def answer(seq):
    queue = 0          # passengers waiting in line (unbounded)
    at_counter = False  # is someone being served at the desk
    belt = 0           # bags waiting on the belt (unbounded)
    for tok in seq:
        if tok == "pax":
            queue += 1
        elif tok == "call":
            if queue > 0:
                queue -= 1
                at_counter = True
            else:
                at_counter = False
        elif tok == "bag":
            if at_counter:
                belt += 1
        elif tok == "load":
            if belt > 0:
                belt -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if queue > belt:
        return "queue"
    if belt > queue:
        return "belt"
    return "even"
