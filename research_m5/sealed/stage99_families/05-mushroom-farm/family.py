"""Mushroom farm grow room.

The room starts dry (humidity 0) with no blocks on the shelf and nothing
harvested. "spawn" puts one new colonized block on the shelf. "mist" raises
the humidity by one, up to a maximum of 2; misting when the humidity is
already 2 brings mold instead, which destroys one block on the shelf (if there
is any) and leaves the humidity at 2. "fan" lowers the humidity by one (never
below 0). "pick" harvests one block: if the humidity is at least 1 and the
shelf holds at least one block, that block is removed from the shelf and the
harvest grows by one; otherwise "pick" does nothing. The answer is the total
number of blocks harvested over the whole sequence, written as a decimal
number (for example "3").
"""

ALPHABET = ["spawn", "mist", "fan", "pick"]


def answer(seq):
    humidity = 0
    blocks = 0
    harvested = 0
    for tok in seq:
        if tok == "spawn":
            blocks += 1
        elif tok == "mist":
            if humidity < 2:
                humidity += 1
            elif blocks > 0:
                blocks -= 1
        elif tok == "fan":
            if humidity > 0:
                humidity -= 1
        elif tok == "pick":
            if humidity >= 1 and blocks > 0:
                blocks -= 1
                harvested += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(harvested)
