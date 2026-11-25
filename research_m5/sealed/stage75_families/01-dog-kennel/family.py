"""A small dog kennel with three pens, watched by one keeper.

The kennel starts empty and calm. Each token is one event of the day: "drop"
means an owner drops off a dog (it gets a pen if one of the three is free,
otherwise it is turned away); "pick" means an owner collects a dog (if any are
there; when the last dog leaves, the empty kennel falls calm); "bark" means a
bark is heard, which raises the noise one step (calm -> restless -> howling,
and it stays at howling) but only if at least one dog is in the kennel, since
an empty kennel cannot bark; "treat" means the keeper hands out treats, which
makes the kennel calm again. The answer is the number of dogs in the kennel at
the end of the day and its noise, written like "2 dogs, restless".
"""

ALPHABET = ["drop", "pick", "bark", "treat"]

_CAPACITY = 3
_NOISE = ["calm", "restless", "howling"]


def answer(seq):
    dogs = 0
    noise = 0
    for tok in seq:
        if tok == "drop":
            if dogs < _CAPACITY:
                dogs += 1
        elif tok == "pick":
            if dogs > 0:
                dogs -= 1
                if dogs == 0:
                    noise = 0
        elif tok == "bark":
            if dogs > 0 and noise < 2:
                noise += 1
        elif tok == "treat":
            noise = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%d dogs, %s" % (dogs, _NOISE[noise])
