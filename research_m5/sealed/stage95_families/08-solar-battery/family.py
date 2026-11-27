"""Solar panels with a home battery: the electricity bill.

A house has rooftop solar panels and a home battery that holds at most 3
units and starts empty; the household's bill starts at 0. Read the tokens
in order, one unit at a time. "sun" produces 3 units and "cloud" produces
1 unit: each produced unit goes into the battery if it holds fewer than 3,
otherwise it is sold to the grid and lowers the bill by 1. "lamp" needs 1
unit and "oven" needs 2 units: each needed unit is taken from the battery
if it holds at least one, otherwise it is bought from the grid and raises
the bill by 2. After the last token the answer is "owe" if the bill is
above 0, "credit" if it is below 0, and "even" if it is exactly 0.
"""

ALPHABET = ["sun", "cloud", "lamp", "oven"]

_CAPACITY = 3
_PRODUCE = {"sun": 3, "cloud": 1}
_NEED = {"lamp": 1, "oven": 2}
_BUY_PRICE = 2
_SELL_PRICE = 1


def answer(seq):
    battery = 0
    bill = 0
    for token in seq:
        if token in _PRODUCE:
            for _ in range(_PRODUCE[token]):
                if battery < _CAPACITY:
                    battery += 1
                else:
                    bill -= _SELL_PRICE
        elif token in _NEED:
            for _ in range(_NEED[token]):
                if battery > 0:
                    battery -= 1
                else:
                    bill += _BUY_PRICE
        else:
            raise ValueError("unknown token: %r" % (token,))
    if bill > 0:
        return "owe"
    if bill < 0:
        return "credit"
    return "even"
