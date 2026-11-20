"""Ice rink: counting falls on worn ice.

The rink opens with freshly resurfaced ice (wear 0). Each token is one event on
the ice. "glide" (a lap of public skating) adds 1 to the wear, "hockey" (a
scrimmage) adds 2, and "jump" (a figure skater's jump) adds 1; wear never goes
above 3. "zamboni" resurfaces the ice and sets the wear back to 0. A jump lands
badly and counts as a fall when the ice, just before that jump, has wear 2 or
more (the jump's own wear is added afterwards). The answer is the total number
of falls over the whole session, written as a decimal string ("0", "1", ...).
"""

ALPHABET = ["glide", "jump", "hockey", "zamboni"]

_WEAR_ADDED = {"glide": 1, "jump": 1, "hockey": 2}
_MAX_WEAR = 3


def answer(seq):
    wear = 0
    falls = 0
    for tok in seq:
        if tok == "zamboni":
            wear = 0
            continue
        if tok == "jump" and wear >= 2:
            falls += 1
        wear = min(_MAX_WEAR, wear + _WEAR_ADDED[tok])
    return str(falls)
