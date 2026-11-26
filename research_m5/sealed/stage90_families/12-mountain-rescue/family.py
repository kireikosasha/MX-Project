"""Mountain rescue base: how many emergency calls are still waiting at the end.

The base starts with clear weather, no calls waiting and no crews standing by.
Read the tokens in order. "call": an emergency call comes in; if the weather is
clear and at least one crew is standing by, one standing-by crew goes out and
answers it at once, otherwise the call joins the waiting calls. "crew": a crew
reports ready at the base; if the weather is clear and at least one call is
waiting, the crew answers one waiting call at once, otherwise it joins the
crews standing by. "storm": the weather turns bad (nobody can go out; calls and
crews just pile up). "clear": the weather turns good, and immediately as many
waiting calls as possible are answered, one standing-by crew per call (both
numbers drop by the smaller of the two). Repeated "storm" or "clear" tokens
change nothing more. The answer is the number of calls still waiting after the
last token, written as a decimal string (e.g. "0", "3").
"""

ALPHABET = ["call", "crew", "storm", "clear"]


def answer(seq):
    stormy = False
    waiting = 0   # calls not yet answered
    standby = 0   # crews ready at base
    for tok in seq:
        if tok == "call":
            if not stormy and standby > 0:
                standby -= 1
            else:
                waiting += 1
        elif tok == "crew":
            if not stormy and waiting > 0:
                waiting -= 1
            else:
                standby += 1
        elif tok == "storm":
            stormy = True
        elif tok == "clear":
            stormy = False
            m = min(waiting, standby)
            waiting -= m
            standby -= m
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(waiting)
