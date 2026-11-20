"""Post-office counter: the stamp drawer and the unpaid tray.

A clerk works a counter with a stamp drawer that holds at most 3 stamps
(it starts empty) and an unpaid tray (it starts empty). Reading the tokens
in order: "stamp" puts one stamp in the drawer, unless the drawer already
holds 3, in which case the stamp is wasted; "letter" needs 1 stamp: if the
drawer has at least 1, one stamp is taken out and the letter is posted,
otherwise the letter goes into the unpaid tray (no stamp is taken); "parcel"
needs 2 stamps: if the drawer has at least 2, two stamps are taken out and
the parcel is posted, otherwise the parcel goes into the unpaid tray (no
stamp is taken); "van" is the mail van calling, which returns everything in
the unpaid tray to sender, leaving the tray empty (the drawer is untouched).
The answer is the number of stamps left in the drawer at the end, a hyphen,
and "held" if the unpaid tray holds anything at the end or "clear" if it is
empty, e.g. "2-clear" or "0-held".
"""

ALPHABET = ["stamp", "letter", "parcel", "van"]


def answer(seq):
    drawer = 0
    tray_full = False
    for tok in seq:
        if tok == "stamp":
            if drawer < 3:
                drawer += 1
        elif tok == "letter":
            if drawer >= 1:
                drawer -= 1
            else:
                tray_full = True
        elif tok == "parcel":
            if drawer >= 2:
                drawer -= 2
            else:
                tray_full = True
        elif tok == "van":
            tray_full = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%d-%s" % (drawer, "held" if tray_full else "clear")
