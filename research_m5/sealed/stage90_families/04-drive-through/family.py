"""Drive-through closing report: did the lane lose more customers or waste more food?

Read the tokens left to right while keeping three numbers in mind: the cars
waiting in the lane, the finished bags on the warming shelf, and a running
balance (customers lost minus bags wasted); all start at 0 and the lane and
shelf have no size limit.  "car": a car joins the lane.  "fry": the kitchen
puts one more bag on the shelf.  "hand": if a car is waiting AND a bag is on
the shelf, the front car takes a bag and leaves (cars -1, bags -1); otherwise
nothing happens.  "honk": if a car is waiting and the shelf is EMPTY, the front
driver gives up and drives off (cars -1, balance +1); otherwise nothing.
"toss": if a bag is on the shelf and NO car is waiting, one cold bag goes in
the bin (bags -1, balance -1); otherwise nothing.  At the end the answer is
"lost" if the balance is positive (more customers lost than bags wasted),
"waste" if it is negative, and "even" if it is zero.
"""

ALPHABET = ["car", "fry", "hand", "honk", "toss"]


def answer(seq):
    cars = 0
    bags = 0
    balance = 0  # customers lost minus bags wasted
    for tok in seq:
        if tok == "car":
            cars += 1
        elif tok == "fry":
            bags += 1
        elif tok == "hand":
            if cars > 0 and bags > 0:
                cars -= 1
                bags -= 1
        elif tok == "honk":
            if cars > 0 and bags == 0:
                cars -= 1
                balance += 1
        elif tok == "toss":
            if bags > 0 and cars == 0:
                bags -= 1
                balance -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "lost"
    if balance < 0:
        return "waste"
    return "even"
