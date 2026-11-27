"""Cannery packing line at the end of a shift.

Tokens arrive one at a time on the packing line. "fish" puts one fish on the
packing table. "can" sends one empty can down the line: if at least one fish is
waiting on the table, one fish is packed into it and the can joins the row of
open (filled but unsealed) cans; if the table is empty, the empty can simply
rolls past and nothing changes. "lid" brings one lid: if at least one open
filled can is waiting, one of them is sealed and leaves the line; otherwise the
lid is thrown away and nothing changes. Both the fish on the table and the open
cans can pile up without limit. After the whole sequence, the answer compares
the two piles: "fish" if more fish are still waiting on the table than there
are open cans, "cans" if more open cans are waiting than fish, and "even" if
the two numbers are equal.
"""

ALPHABET = ["fish", "can", "lid"]


def answer(seq):
    table = 0       # fish waiting on the packing table
    open_cans = 0   # filled cans waiting for a lid
    for tok in seq:
        if tok == "fish":
            table += 1
        elif tok == "can":
            if table > 0:
                table -= 1
                open_cans += 1
        elif tok == "lid":
            if open_cans > 0:
                open_cans -= 1
    if table > open_cans:
        return "fish"
    if open_cans > table:
        return "cans"
    return "even"
