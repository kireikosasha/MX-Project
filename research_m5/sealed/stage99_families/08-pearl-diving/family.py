"""Pearl-diving boat.

A pearl-diving boat starts the day with nobody in the water and an empty basket.
Read the tokens in order. "dive": one more diver goes down from the boat. "pearl":
an oyster with a pearl is found below; it is taken by the diver who has been down
the longest among those whose hands are still empty, and if no diver is below or
every diver below already holds a pearl, it is left on the seabed (each diver can
hold only one pearl). "rise": the diver who has been down the longest climbs back
into the boat, and if he holds a pearl it goes into the basket; if nobody is below,
nothing happens. "shark": every diver below rushes back to the boat at once,
dropping whatever pearls they hold, so nothing reaches the basket and nobody is
left in the water. The answer is the number of pearls in the basket at the end of
the sequence, written as a decimal string (pearls still held underwater at the end
do not count).
"""

ALPHABET = ["dive", "pearl", "rise", "shark"]


def answer(seq):
    below = 0     # divers in the water
    holding = 0   # how many of them hold a pearl (always the longest-down ones)
    basket = 0    # pearls safely in the boat
    for tok in seq:
        if tok == "dive":
            below += 1
        elif tok == "pearl":
            if holding < below:
                holding += 1
        elif tok == "rise":
            if below > 0:
                below -= 1
                if holding > 0:
                    holding -= 1
                    basket += 1
        elif tok == "shark":
            below = 0
            holding = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(basket)
