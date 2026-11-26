"""Penalty box of an ice hockey game.

Read the tokens in order while keeping three numbers: how many home players
sit in the penalty box, how many away players sit in it (both start at 0 and
can grow without limit), and the score difference (home goals minus away
goals, starting at 0). "home" sends one home player to the box; "away" sends
one away player to the box; "clock" means two minutes pass, so each box that
is not empty lets one of its players out (a box already empty stays at 0);
"goal" means a goal is scored by the team with fewer players in the box (the
power-play team): if the home box holds fewer players than the away box, home
scores and one away player leaves the box; if the away box holds fewer, away
scores and one home player leaves the box; if both boxes hold the same number
of players (even strength, including both empty), the goal does not count and
nothing changes. The answer is who is ahead after the last token: "home" if
the score difference is positive, "away" if it is negative, "tie" if it is 0.
"""

ALPHABET = ["home", "away", "goal", "clock"]


def answer(seq):
    home_box = 0
    away_box = 0
    diff = 0  # home goals minus away goals
    for tok in seq:
        if tok == "home":
            home_box += 1
        elif tok == "away":
            away_box += 1
        elif tok == "clock":
            if home_box > 0:
                home_box -= 1
            if away_box > 0:
                away_box -= 1
        elif tok == "goal":
            if home_box < away_box:
                diff += 1
                away_box -= 1
            elif away_box < home_box:
                diff -= 1
                home_box -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if diff > 0:
        return "home"
    if diff < 0:
        return "away"
    return "tie"
