"""Sparring scorecard at a boxing gym.

Two boxers, red and blue, spar while a coach calls out what happens. Red starts
as the one pressing forward. A "jab" scores 1 point for whoever is currently
pressing, a "hook" scores 2 points for whoever is currently pressing, and a
"slip" means the other boxer slipped the punch and is now the one pressing
(points are unchanged; who presses carries over across rounds). A "bell" ends
the current round: the boxer with more points in that round wins it (equal
points means nobody wins it), and the round points go back to zero for the next
round. When the sequence ends, the round in progress is scored as if the bell
had rung. The answer is the boxer who won more rounds, "red" or "blue", or
"draw" if they won the same number of rounds.
"""

ALPHABET = ["jab", "hook", "slip", "bell"]


def answer(seq):
    pressing = 0  # 0 = red, 1 = blue
    points = [0, 0]
    rounds = [0, 0]

    def close_round():
        if points[0] > points[1]:
            rounds[0] += 1
        elif points[1] > points[0]:
            rounds[1] += 1
        points[0] = points[1] = 0

    for tok in seq:
        if tok == "jab":
            points[pressing] += 1
        elif tok == "hook":
            points[pressing] += 2
        elif tok == "slip":
            pressing = 1 - pressing
        elif tok == "bell":
            close_round()
    close_round()

    if rounds[0] > rounds[1]:
        return "red"
    if rounds[1] > rounds[0]:
        return "blue"
    return "draw"
