"""Polling station with one booth and a waiting line.

Read the tokens in order. "voter" means a new voter arrives and joins the end
of the line. "red" and "blue" mean that a ballot for that party is cast by the
voter at the front of the line, who then leaves the line; "spoil" means the
voter at the front spoils their ballot and leaves without a vote being counted.
If the line is empty when "red", "blue" or "spoil" is read, nobody is there to
cast it and the token is ignored (it changes nothing). At the end, compare the
number of counted red votes with the number of counted blue votes: the answer
is "red" if red has more, "blue" if blue has more, and "tie" if they are equal
(including when no votes were counted). Voters still waiting in line at the end
do not affect the answer.
"""

ALPHABET = ["voter", "red", "blue", "spoil"]


def answer(seq):
    waiting = 0
    margin = 0  # counted red votes minus counted blue votes
    for tok in seq:
        if tok == "voter":
            waiting += 1
        elif waiting > 0:
            waiting -= 1
            if tok == "red":
                margin += 1
            elif tok == "blue":
                margin -= 1
    if margin > 0:
        return "red"
    if margin < 0:
        return "blue"
    return "tie"
