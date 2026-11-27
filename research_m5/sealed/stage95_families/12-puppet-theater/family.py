"""Puppet theater: the curtain starts closed and there are no puppets on the
stage and none in the wings. Read the tokens in order. "tug" pulls the curtain
rope, toggling it between closed and open. "up" lifts a new puppet: it appears
on the stage if the curtain is open, otherwise it waits in the wings. "down"
lowers one puppet away for good: with the curtain open it takes one from the
stage, with the curtain closed it takes one from the wings (if that place is
empty, nothing happens). "scene" is a scene change, whatever the curtain: all
puppets on the stage go to the wings and, at the same time, all puppets that
were in the wings come onto the stage. After the last token, answer "stage"
if more puppets are on the stage than in the wings, "wings" if more are in the
wings, and "tie" if the two counts are equal.
"""

ALPHABET = ["tug", "up", "down", "scene"]


def answer(seq):
    curtain_open = False
    stage = 0
    wings = 0
    for tok in seq:
        if tok == "tug":
            curtain_open = not curtain_open
        elif tok == "up":
            if curtain_open:
                stage += 1
            else:
                wings += 1
        elif tok == "down":
            if curtain_open:
                if stage > 0:
                    stage -= 1
            else:
                if wings > 0:
                    wings -= 1
        elif tok == "scene":
            stage, wings = wings, stage
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if stage > wings:
        return "stage"
    if wings > stage:
        return "wings"
    return "tie"
