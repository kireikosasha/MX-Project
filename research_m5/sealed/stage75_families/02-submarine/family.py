"""A submarine's dive log.

The boat starts at the surface with its hatch closed. Depth has four levels:
surface, periscope, cruise, deep. "dive" takes the boat one level deeper
(it stays at deep if already there) but does nothing while the hatch is open;
"rise" brings it one level up (it stays at the surface if already there);
"blow" is an emergency blow of the ballast tanks that brings it straight to
the surface from any depth; "hatch" opens the hatch if it is closed and closes
it if it is open, but only when the boat is at the surface -- under water the
crew ignores the order. The answer is where the boat is after the whole log:
"open" if it is at the surface with the hatch open, otherwise the name of its
depth level ("surface", "periscope", "cruise" or "deep").
"""

ALPHABET = ["dive", "rise", "blow", "hatch"]

_LEVELS = ["surface", "periscope", "cruise", "deep"]


def answer(seq):
    depth = 0
    hatch_open = False
    for tok in seq:
        if tok == "dive":
            if not hatch_open and depth < 3:
                depth += 1
        elif tok == "rise":
            if depth > 0:
                depth -= 1
        elif tok == "blow":
            depth = 0
        elif tok == "hatch":
            if depth == 0:
                hatch_open = not hatch_open
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if hatch_open:
        return "open"
    return _LEVELS[depth]
