"""Weather station precipitation log.

The station's thermometer has five levels, 0 (freezing) to 4 (hot), and starts
at level 2 (mild). Reading the log token by token: "warm" raises the
thermometer one level (it cannot go above 4), "cool" lowers it one level (it
cannot go below 0), and "calm" resets it to level 2. Each "rain" token is a
precipitation event and does not change the thermometer: if the thermometer is
at level 0 or 1 at that moment it is recorded as snowfall, if it is at level 3
or 4 it is recorded as rainfall, and at level 2 it is sleet, which is recorded
as neither. The answer compares the totals at the end of the log: "snow" if
there were more snowfall records than rainfall records, "rain" if there were
more rainfall records than snowfall records, and "even" if the two totals are
equal (including when nothing was recorded at all).
"""

ALPHABET = ["warm", "cool", "rain", "calm"]


def answer(seq):
    level = 2
    balance = 0  # rainfall records minus snowfall records
    for tok in seq:
        if tok == "warm":
            if level < 4:
                level += 1
        elif tok == "cool":
            if level > 0:
                level -= 1
        elif tok == "calm":
            level = 2
        elif tok == "rain":
            if level <= 1:
                balance -= 1
            elif level >= 3:
                balance += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "rain"
    if balance < 0:
        return "snow"
    return "even"
