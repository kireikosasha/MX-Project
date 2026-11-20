"""Dentist family: the state of a patient's teeth after a stretch of daily life.

The patient starts with clean teeth (plaque 0, no cavity). Reading the tokens in
order: "candy" adds 1 plaque, "soda" adds 2 plaque, "brush" removes 1 plaque
(never below 0), and "checkup" is a dentist visit that fills any open cavity and
scrapes the plaque back to 0. Whenever the plaque reaches 4 or more, it eats into
a tooth: the patient now has an open cavity (it stays open until the next
checkup; extra cavities do not change anything) and the plaque drops back to 0.
The answer describes the mouth at the end: "cavity" if a cavity is open,
otherwise "plaque" if the plaque is 2 or more, otherwise "clean".
"""

ALPHABET = ["candy", "soda", "brush", "checkup"]


def answer(seq):
    plaque = 0
    cavity = False
    for tok in seq:
        if tok == "candy":
            plaque += 1
        elif tok == "soda":
            plaque += 2
        elif tok == "brush":
            if plaque > 0:
                plaque -= 1
        elif tok == "checkup":
            cavity = False
            plaque = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
        if plaque >= 4:
            cavity = True
            plaque = 0
    if cavity:
        return "cavity"
    if plaque >= 2:
        return "plaque"
    return "clean"
