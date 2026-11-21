"""Lion-tamer act.

A lion in the ring has a mood (calm < edgy < wild) and a perch (floor or
stool); the act starts with the lion calm and on the floor.  Each token is
something that happens in the ring: "drum" (a drum roll) raises the mood one
step (wild stays wild); "meat" (a treat) lowers the mood one step (calm stays
calm); "whip" (a crack of the whip) sends the lion up onto its stool unless it
is wild, in which case a wild lion leaps down to the floor instead (its mood
stays wild); "hoop" (a flaming hoop is held up) makes a lion on the stool leap
through it and land on the floor with its mood unchanged, while a lion already
on the floor is frightened by the hoop and its mood rises one step.  The answer
is the lion's mood and perch after the last token, written "mood-perch", for
example "edgy-stool".
"""

ALPHABET = ["drum", "meat", "whip", "hoop"]

_MOODS = ["calm", "edgy", "wild"]


def answer(seq):
    mood = 0          # index into _MOODS
    on_stool = False  # start on the floor
    for tok in seq:
        if tok == "drum":
            if mood < 2:
                mood += 1
        elif tok == "meat":
            if mood > 0:
                mood -= 1
        elif tok == "whip":
            if mood == 2:
                on_stool = False
            else:
                on_stool = True
        elif tok == "hoop":
            if on_stool:
                on_stool = False
            elif mood < 2:
                mood += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _MOODS[mood] + "-" + ("stool" if on_stool else "floor")
