"""Carousel riders.

A carousel has five horses arranged in a ring, numbered 0 to 4, all empty at
the start; the boarding gate faces horse 0 and the carousel is set to turn
forward (horse number increasing). Read the tokens in order: "spin" turns the
carousel one step in its current direction, so the gate now faces the next
horse (forward: k -> k+1, backward: k -> k-1, wrapping around mod 5);
"reverse" flips the turning direction without moving; "board" seats a rider on
the horse at the gate if it is empty (otherwise nothing happens); "alight"
lets the rider on the horse at the gate get off if it is occupied (otherwise
nothing happens). The answer is the number of riders on the carousel after the
last token, written as a decimal digit string ("0" to "5").
"""

ALPHABET = ["spin", "reverse", "board", "alight"]

N_HORSES = 5


def answer(seq):
    gate = 0
    step = 1
    seated = [False] * N_HORSES
    for tok in seq:
        if tok == "spin":
            gate = (gate + step) % N_HORSES
        elif tok == "reverse":
            step = -step
        elif tok == "board":
            seated[gate] = True
        elif tok == "alight":
            seated[gate] = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sum(seated))
