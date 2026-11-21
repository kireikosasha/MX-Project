"""Gas station with one underground tank.

The station's tank holds at most 6 units of fuel and starts full. Tokens are
read in order: a "car" wants 1 unit and a "truck" wants 3 units; if the tank
holds at least that much, the vehicle is served and the fuel is removed,
otherwise the vehicle drives off unserved and the tank is left unchanged. A
"tanker" refills the tank to 6 units. The answer is the number of vehicles
(cars and trucks) that drove off unserved, written as a decimal string.
"""

ALPHABET = ["car", "truck", "tanker"]

CAPACITY = 6
NEED = {"car": 1, "truck": 3}


def answer(seq):
    fuel = CAPACITY
    unserved = 0
    for tok in seq:
        if tok == "tanker":
            fuel = CAPACITY
        else:
            need = NEED[tok]
            if fuel >= need:
                fuel -= need
            else:
                unserved += 1
    return str(unserved)
