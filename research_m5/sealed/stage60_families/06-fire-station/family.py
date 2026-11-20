"""Fire station engine bay.

The station owns three engines and starts the day with all three parked in the
bay. Read the tokens in order: "alarm" sends one engine out (if at least one is
parked), "blaze" sends two engines out (or as many as are parked, if fewer than
two), "back" brings one engine home (if at least one is out), and "drill" changes
nothing because the crew trains in the yard. Requests that cannot be met are
simply ignored, so the bay never holds fewer than zero or more than three
engines. The answer is how many engines are parked in the bay after the last
token: "none home", "one home", "two home" or "all home".
"""

ALPHABET = ["alarm", "blaze", "back", "drill"]

_CAPACITY = 3
_WORDS = ["none home", "one home", "two home", "all home"]


def answer(seq):
    home = _CAPACITY
    for tok in seq:
        if tok == "alarm":
            home = max(0, home - 1)
        elif tok == "blaze":
            home = max(0, home - 2)
        elif tok == "back":
            home = min(_CAPACITY, home + 1)
        elif tok == "drill":
            pass
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _WORDS[home]
