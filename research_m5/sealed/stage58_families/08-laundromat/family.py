"""Laundromat basket.

One basket of laundry moves through a single coin-operated washer and dryer.
The basket starts "dirty" and the coin slot starts empty; the slot holds at
most 3 coins. Read the tokens left to right: "coin" puts one coin in the slot
(ignored if the slot already holds 3); "wash" runs the washer only if the slot
holds at least 2 coins, spending 2 and leaving the basket "wet" whatever state
it was in (otherwise nothing happens); "dry" runs the dryer only if the basket
is "wet" and the slot holds at least 1 coin, spending 1 and leaving the basket
"dry" (otherwise nothing happens); "fold" only does something when the basket
is "dry": the folded load is handed back and a fresh "dirty" basket takes its
place. The answer is the basket's state after the last token: "dirty", "wet"
or "dry".
"""

ALPHABET = ["coin", "wash", "dry", "fold"]


def answer(seq):
    coins = 0
    stage = "dirty"
    for tok in seq:
        if tok == "coin":
            if coins < 3:
                coins += 1
        elif tok == "wash":
            if coins >= 2:
                coins -= 2
                stage = "wet"
        elif tok == "dry":
            if stage == "wet" and coins >= 1:
                coins -= 1
                stage = "dry"
        elif tok == "fold":
            if stage == "dry":
                stage = "dirty"
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return stage
