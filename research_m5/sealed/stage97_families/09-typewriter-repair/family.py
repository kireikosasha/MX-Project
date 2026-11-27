"""Typewriter repair shop: complaints from customers who come too early.

The shop opens with nothing in it and reads the day's events in order.
"drop" means a customer leaves a broken typewriter, which goes on the
repair bench. "ribbon" means one new ribbon is delivered to the stock
room. "fix" means the repairer tries to repair one typewriter: this works
only if at least one typewriter is on the bench and at least one ribbon is
in stock, and then one typewriter leaves the bench, one ribbon is used up,
and the repaired typewriter goes on the pick-up shelf; otherwise nothing
happens. "collect" means a customer comes to collect a typewriter: if the
pick-up shelf has at least one, the customer takes one away; if the shelf
is empty, the customer leaves without one and files a complaint. The
answer is the total number of complaints at closing time, written as a
decimal string (for example "0" or "3").
"""

ALPHABET = ["drop", "ribbon", "fix", "collect"]


def answer(seq):
    bench = 0
    ribbons = 0
    shelf = 0
    complaints = 0
    for tok in seq:
        if tok == "drop":
            bench += 1
        elif tok == "ribbon":
            ribbons += 1
        elif tok == "fix":
            if bench > 0 and ribbons > 0:
                bench -= 1
                ribbons -= 1
                shelf += 1
        elif tok == "collect":
            if shelf > 0:
                shelf -= 1
            else:
                complaints += 1
    return str(complaints)
