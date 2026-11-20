"""Ski-lift loading line.

The sequence is read as events at the bottom station of a double chairlift. The
lift starts running and the loading line starts empty. "skier" means one skier
joins the line. "chair" means a chair swings through the loading point: if the
lift is running it carries away up to two skiers from the line (fewer if fewer
are waiting); if the lift is on wind hold the chair does not come and nothing
happens. "gust" puts the lift on wind hold, and "calm" sets it running again
(repeated gusts or calms change nothing further). The answer is the number of
skiers still waiting in the line after the last event, written as a decimal
string such as "0" or "7".
"""

ALPHABET = ["skier", "chair", "gust", "calm"]


def answer(seq):
    running = True
    waiting = 0
    for tok in seq:
        if tok == "skier":
            waiting += 1
        elif tok == "chair":
            if running:
                waiting = max(0, waiting - 2)
        elif tok == "gust":
            running = False
        elif tok == "calm":
            running = True
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(waiting)
