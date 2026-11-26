"""Call center shift tally.

Read the tokens in order as events at a customer call center whose hold
queue starts empty. "ring" means a new caller dials in and is put on hold
(the hold queue grows by one, with no upper limit). "pickup" means an agent
takes one caller off hold and serves them; "hangup" means one caller on hold
loses patience and leaves, counting as abandoned; both do nothing at all if
nobody is on hold. "rush" means the supervisor pulls in extra staff: every
caller currently on hold is served at once and the queue becomes empty.
"outage" means the phone line drops: every caller currently on hold is lost
at once, each counting as abandoned, and the queue becomes empty. ("rush" and
"outage" do nothing if nobody is on hold.) After the whole sequence, the
answer is "served" if more callers were served than abandoned in total,
"abandoned" if more were abandoned than served, and "even" if the two totals
are equal.
"""

ALPHABET = ["ring", "pickup", "hangup", "rush", "outage"]


def answer(seq):
    on_hold = 0
    balance = 0  # callers served minus callers abandoned
    for tok in seq:
        if tok == "ring":
            on_hold += 1
        elif tok == "pickup":
            if on_hold:
                on_hold -= 1
                balance += 1
        elif tok == "hangup":
            if on_hold:
                on_hold -= 1
                balance -= 1
        elif tok == "rush":
            balance += on_hold
            on_hold = 0
        elif tok == "outage":
            balance -= on_hold
            on_hold = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "served"
    if balance < 0:
        return "abandoned"
    return "even"
