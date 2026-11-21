"""Zoo feeding round: the mood of the zoo after the keeper's walk.

A keeper walks the zoo carrying a single bucket, which starts empty, and the
zoo's noise level starts at 0 (it can range from 0 to 3). Read the tokens in
order. "fish" or "meat" refills the bucket with that food, throwing out
whatever was in it before. "rain" spoils the bucket, leaving it empty. "seal"
is a seal coming up to be fed: if the bucket holds fish, the seal eats it (the
bucket becomes empty) and the noise level drops by 1 (never below 0);
otherwise the seal goes hungry and the noise level rises by 1 (never above 3).
"lion" works the same way but needs meat. The answer is the noise level at the
end, named "calm" (0), "restless" (1), "noisy" (2) or "uproar" (3).
"""

ALPHABET = ["fish", "meat", "seal", "lion", "rain"]

_LEVELS = ["calm", "restless", "noisy", "uproar"]
_DIET = {"seal": "fish", "lion": "meat"}


def answer(seq):
    bucket = None
    noise = 0
    for tok in seq:
        if tok in ("fish", "meat"):
            bucket = tok
        elif tok == "rain":
            bucket = None
        elif tok in _DIET:
            if bucket == _DIET[tok]:
                bucket = None
                noise = max(0, noise - 1)
            else:
                noise = min(3, noise + 1)
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _LEVELS[noise]
