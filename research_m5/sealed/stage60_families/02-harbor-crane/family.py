"""Harbor crane unloading a ship onto a quay stack.

The crane's boom starts over the ship with an empty hook and an empty quay
stack. "slew" swings the boom to the other side (ship <-> quay). "hoist"
does nothing if the hook already holds a container; otherwise, over the ship
it picks up a new container, and over the quay it takes the top container
off the quay stack if the stack is not empty. "lower" does nothing if the
hook is empty; otherwise it sets the container down where the boom is: over
the ship it goes back aboard, over the quay it goes onto the stack, and as
soon as the stack reaches three containers a truck carries all three away
(the stack becomes empty). "gust" raises a wind alarm: the very next command
(whatever it is, even another "gust") is ignored, then work resumes. The
answer is the number of containers left on the quay stack at the end, as a
string "0", "1" or "2".
"""

ALPHABET = ["hoist", "lower", "slew", "gust"]


def answer(seq):
    over_quay = False
    loaded = False
    stack = 0
    paused = False
    for tok in seq:
        if paused:
            paused = False
            continue
        if tok == "gust":
            paused = True
        elif tok == "slew":
            over_quay = not over_quay
        elif tok == "hoist":
            if not loaded:
                if not over_quay:
                    loaded = True
                elif stack > 0:
                    stack -= 1
                    loaded = True
        elif tok == "lower":
            if loaded:
                loaded = False
                if over_quay:
                    stack += 1
                    if stack == 3:
                        stack = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(stack)
