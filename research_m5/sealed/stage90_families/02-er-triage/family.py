"""Emergency-room triage at the end of a shift.

The waiting room starts empty and the tokens are read one by one. "urgent"
means an urgent patient arrives and joins the urgent line; "minor" means a
patient with a minor complaint arrives and joins the minor line. "doctor"
means a doctor becomes free and sees one waiting patient: an urgent one if
any urgent patient is waiting, otherwise a minor one if any is waiting,
otherwise nobody. "nurse" means a nurse becomes free and sees one minor
patient if any is waiting; nurses never take urgent patients, so with no
minor patient waiting the nurse sees nobody. Both lines can grow without
limit. The answer describes the waiting room after the last token: "empty"
if nobody is waiting, otherwise "urgent" if more urgent than minor patients
are waiting, "minor" if more minor than urgent patients are waiting, and
"tie" if the two lines are equally long.
"""

ALPHABET = ["urgent", "minor", "doctor", "nurse"]


def answer(seq):
    urgent = 0
    minor = 0
    for tok in seq:
        if tok == "urgent":
            urgent += 1
        elif tok == "minor":
            minor += 1
        elif tok == "doctor":
            if urgent > 0:
                urgent -= 1
            elif minor > 0:
                minor -= 1
        elif tok == "nurse":
            if minor > 0:
                minor -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if urgent == 0 and minor == 0:
        return "empty"
    if urgent > minor:
        return "urgent"
    if minor > urgent:
        return "minor"
    return "tie"
