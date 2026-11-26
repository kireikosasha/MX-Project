"""Space station hull watch.

The station starts with nobody aboard and no open leaks, and the tokens are
read in order: "board" brings one more crew member aboard; "leave" sends one
crew member home (nothing happens if nobody is aboard); "leak" opens one more
hull leak; "patch" closes one open leak, but only if at least one crew member
is aboard to do the work and there is a leak to close (otherwise nothing
happens). Crew aboard and open leaks are both plain counts with no upper
limit. After the last token the answer is "sealed" if no leak is open;
otherwise "abandoned" if nobody is aboard; otherwise "overrun" if open leaks
outnumber the crew aboard; otherwise "holding".
"""

ALPHABET = ["board", "leave", "leak", "patch"]


def answer(seq):
    crew = 0
    leaks = 0
    for tok in seq:
        if tok == "board":
            crew += 1
        elif tok == "leave":
            if crew > 0:
                crew -= 1
        elif tok == "leak":
            leaks += 1
        elif tok == "patch":
            if crew > 0 and leaks > 0:
                leaks -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if leaks == 0:
        return "sealed"
    if crew == 0:
        return "abandoned"
    if leaks > crew:
        return "overrun"
    return "holding"
