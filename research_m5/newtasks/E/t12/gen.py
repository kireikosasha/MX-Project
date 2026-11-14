import random
import sys

# Task: a shared meeting has a day of the week. Some statements set the day
# directly, others shift it by a number of days (modulo the 7-day week).
# Questions ask for the current day or check whether it is a given day, so the
# answer needs the whole chain of statements above it.

NAMES = ["ann", "bob", "cy", "dee", "eve"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUMS = {2: "two", 3: "three", 4: "four", 5: "five"}


def story(rng):
    n_st = rng.randint(3, 12)
    n_q = rng.randint(1, 5)
    total = n_st + n_q
    # question slots are never slot 0, which is always a statement that sets the day
    q_pos = set(rng.sample(range(1, total), n_q))
    day = None
    items = []  # (text, answer or None)
    for i in range(total):
        if i in q_pos:
            if rng.random() < 0.5:
                items.append(("What day is the meeting on?", DAYS[day]))
            else:
                d = day if rng.random() < 0.5 else rng.randrange(7)
                items.append((f"Is the meeting on {DAYS[d]}?", "yes" if d == day else "no"))
        else:
            if day is None or rng.random() < 0.4:
                d = rng.randrange(7)
                if rng.random() < 0.5:
                    text = f"{rng.choice(NAMES)} sets the meeting to {DAYS[d]}."
                else:
                    text = f"The meeting is on {DAYS[d]}."
                day = d
            else:
                n = rng.choice(sorted(NUMS))
                later = rng.random() < 0.5
                day = (day + n) % 7 if later else (day - n) % 7
                text = f"The meeting moves {NUMS[n]} days {'later' if later else 'earlier'}."
            items.append((text, None))
    return items


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    for _ in range(n):
        for k, (text, ans) in enumerate(story(rng), 1):
            if ans is None:
                print(f"{k} {text}")
            else:
                print(f"{k} {text}\t{ans}")


if __name__ == "__main__":
    main()
