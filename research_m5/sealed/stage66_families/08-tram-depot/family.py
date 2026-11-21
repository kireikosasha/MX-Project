"""Tram depot with three parallel tracks.

The depot has three tracks, ordered north, middle, south, all empty at the
start, and a single set of points that starts aimed at the middle track.
Reading the tokens in order: "left" moves the points one track toward north
and "right" one track toward south (at the outermost track the points stay
put); "arrive" parks one tram on the track the points are aimed at; "leave"
sends one tram out of that track (nothing happens if that track is empty).
The answer is the name of the track holding strictly the most trams at the
end ("north", "middle" or "south"), "tie" if two or three tracks share the
largest non-zero count, or "empty" if no trams are left in the depot.
"""

ALPHABET = ["arrive", "leave", "left", "right"]

_TRACKS = ["north", "middle", "south"]


def answer(seq):
    counts = [0, 0, 0]
    pos = 1
    for tok in seq:
        if tok == "left":
            if pos > 0:
                pos -= 1
        elif tok == "right":
            if pos < 2:
                pos += 1
        elif tok == "arrive":
            counts[pos] += 1
        elif tok == "leave":
            if counts[pos] > 0:
                counts[pos] -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    best = max(counts)
    if best == 0:
        return "empty"
    if counts.count(best) > 1:
        return "tie"
    return _TRACKS[counts.index(best)]
