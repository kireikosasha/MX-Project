"""Pharmacy shelf: count the customers turned away.

A pharmacy shelf holds one medicine and has room for at most 4 boxes; at
opening it holds 2 boxes. Read the tokens in order. "restock" fills the shelf
to 4 boxes. "sell" is a customer asking for one box: if the shelf has any box,
one is sold (stock goes down by 1); if the shelf is empty, the customer is
turned away. "return" is a customer bringing a box back, which goes on the
shelf (stock up by 1, but never above 4). "expire" removes the older half of
the shelf as expired (stock becomes stock // 2, rounding down). The answer is
the total number of customers turned away during the whole day, written as a
decimal integer (e.g. "0", "3").
"""

ALPHABET = ["restock", "sell", "return", "expire"]

CAPACITY = 4
START = 2


def answer(seq):
    stock = START
    turned_away = 0
    for tok in seq:
        if tok == "restock":
            stock = CAPACITY
        elif tok == "sell":
            if stock > 0:
                stock -= 1
            else:
                turned_away += 1
        elif tok == "return":
            if stock < CAPACITY:
                stock += 1
        elif tok == "expire":
            stock //= 2
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(turned_away)
