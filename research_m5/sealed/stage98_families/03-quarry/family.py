"""Stone quarry ledger.

The quarry has a yard where cut blocks wait, a cart that holds at most two
blocks, a tally of blocks shipped and a tally of blocks lost to cracks; all
start at zero. Read the tokens in order: "cut" puts one new block on the yard;
"load" moves one block from the yard onto the cart, but only if the yard is
not empty and the cart holds fewer than two blocks (otherwise nothing
happens); "haul" sends the cart off, adding every block on it to the shipped
tally and leaving the cart empty (an empty cart ships nothing); "crack" breaks
one block lying on the yard, removing it from the yard and adding one to the
lost tally (if the yard is empty, nothing happens; blocks on the cart are
safe). After the last token the answer is "shipped" if more blocks were
shipped than lost, "lost" if more were lost than shipped, and "even" if the
two tallies are equal. Blocks still on the yard or on the cart do not count.
"""

ALPHABET = ["cut", "load", "haul", "crack"]

CART_CAPACITY = 2


def answer(seq):
    yard = 0
    cart = 0
    shipped = 0
    lost = 0
    for tok in seq:
        if tok == "cut":
            yard += 1
        elif tok == "load":
            if yard > 0 and cart < CART_CAPACITY:
                yard -= 1
                cart += 1
        elif tok == "haul":
            shipped += cart
            cart = 0
        elif tok == "crack":
            if yard > 0:
                yard -= 1
                lost += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if shipped > lost:
        return "shipped"
    if lost > shipped:
        return "lost"
    return "even"
