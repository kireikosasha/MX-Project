"""Toll-bridge drawbridge till.

A drawbridge with a toll booth starts lowered and with an empty till. Tokens are
read left to right: "lift" toggles the drawbridge (lowered -> raised, raised ->
lowered); a vehicle ("bike", "car", "truck") that arrives while the bridge is
lowered crosses and pays its toll (bike 1, car 2, truck 3 coins), while a vehicle
that arrives while the bridge is raised turns back and pays nothing. The till is
emptied into a strongbox every time it reaches 5 coins, so only the remainder is
visible: the answer is the number of coins left in the till at the end, i.e. the
total toll collected modulo 5, written as a decimal digit string "0" to "4".
"""

ALPHABET = ["bike", "car", "truck", "lift"]

_TOLL = {"bike": 1, "car": 2, "truck": 3}


def answer(seq):
    raised = False
    till = 0
    for tok in seq:
        if tok == "lift":
            raised = not raised
        elif not raised:
            till = (till + _TOLL[tok]) % 5
    return str(till)
