"""Clock-tower hour hand.

The tower's hour hand starts at 12 and its mainspring starts unwound (0 turns
of tension; it holds at most 3). Read the tokens in order: "wind" adds one turn
of tension (extra winding beyond 3 is wasted); "tick" moves the hand forward one
hour on the 12-hour dial, but only if the spring holds at least one turn, and
that tick uses up one turn (with an empty spring a tick does nothing); "gust"
blows the hand back one hour on the dial regardless of the spring; "rust" seizes
the mainspring and drains all of its tension to 0 without moving the hand. The
answer is the hour the hand shows after the last token, as a string "1".."12".
"""

ALPHABET = ["wind", "tick", "gust", "rust"]

_MAX_TENSION = 3


def answer(seq):
    hour = 0          # 0 means 12 o'clock; dial positions 0..11
    tension = 0
    for tok in seq:
        if tok == "wind":
            if tension < _MAX_TENSION:
                tension += 1
        elif tok == "tick":
            if tension > 0:
                tension -= 1
                hour = (hour + 1) % 12
        elif tok == "gust":
            hour = (hour - 1) % 12
        elif tok == "rust":
            tension = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(hour if hour != 0 else 12)
