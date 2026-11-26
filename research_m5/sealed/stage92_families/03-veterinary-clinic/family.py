"""Veterinary clinic waiting room.

The sequence is one day at the clinic, read in order. A "dog" or "cat" token
is an animal of that species arriving and sitting down in the waiting room.
A "bark" token means a waiting dog barks: if at least one dog and at least one
cat are waiting, one frightened cat runs out of the clinic untreated; a "hiss"
token means a waiting cat hisses: if at least one dog and at least one cat are
waiting, one frightened dog runs out untreated (in both cases nothing happens
unless both species are present). A "vet" token means the vet calls in one
patient: the species with more animals currently waiting sends one animal in
to be treated; on a tie with animals present a cat goes in first (cats get
stressed waiting); if the room is empty nothing happens. At closing time the
answer is "dogs" if over the whole day the vet treated more dogs than cats,
"cats" if the vet treated more cats than dogs, and "even" if the two counts
are equal.
"""

ALPHABET = ["dog", "cat", "bark", "hiss", "vet"]


def answer(seq):
    dogs_waiting = 0
    cats_waiting = 0
    balance = 0  # dogs treated minus cats treated
    for tok in seq:
        if tok == "dog":
            dogs_waiting += 1
        elif tok == "cat":
            cats_waiting += 1
        elif tok == "bark":
            if dogs_waiting > 0 and cats_waiting > 0:
                cats_waiting -= 1
        elif tok == "hiss":
            if dogs_waiting > 0 and cats_waiting > 0:
                dogs_waiting -= 1
        elif tok == "vet":
            if dogs_waiting == 0 and cats_waiting == 0:
                continue
            if dogs_waiting > cats_waiting:
                dogs_waiting -= 1
                balance += 1
            else:
                cats_waiting -= 1
                balance -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "dogs"
    if balance < 0:
        return "cats"
    return "even"
