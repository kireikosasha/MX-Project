"""Mine-shaft cage and ore cart.

A cage with a single ore cart works a shaft with four stops: the surface
(level 0) and levels 1, 2 and 3 (the bottom). The cage starts at the surface
with an empty cart. "down" moves the cage one level deeper (it stays put at
level 3) and "up" moves it one level higher (it stays put at the surface).
"dig" fills the cart with the ore of the current level -- coal on level 1,
iron on level 2, gold on level 3 -- but only if the cage is below the surface
and the cart is empty; otherwise it does nothing. "haul" empties a loaded
cart onto the surface stockpile, but only when the cage is at the surface;
otherwise it does nothing. The answer is the kind of ore most recently
emptied onto the stockpile ("coal", "iron" or "gold"), or "none" if no
load has reached the surface yet.
"""

ALPHABET = ["down", "up", "dig", "haul"]

_ORE = {1: "coal", 2: "iron", 3: "gold"}
_BOTTOM = 3


def answer(seq):
    level = 0
    cart = None
    last = "none"
    for tok in seq:
        if tok == "down":
            if level < _BOTTOM:
                level += 1
        elif tok == "up":
            if level > 0:
                level -= 1
        elif tok == "dig":
            if level > 0 and cart is None:
                cart = _ORE[level]
        elif tok == "haul":
            if level == 0 and cart is not None:
                last = cart
                cart = None
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return last
