#!/usr/bin/env python3
"""
The micro-world of quantities: word problems made by a program, so the answer is known exactly and there are
as many as needed.

What it is for: one story with several questions. "Tom has 5 apples. Ann has 7 apples." can be asked for the
total, for the difference, for Ann and Tom but not Bob - and the right expression changes with the question
while the story stays. Collected problems rarely show that; here it is the rule.

Kinds of stories: several owners of one thing, one owner on different days / of different colours, a change
(had - got - lost, with any of the three unknown), a comparison (more than, fewer than), equal groups and
prices (times, divided), groups with two sorts in each, "every day", a whole and its parts, and two steps of
these. Sentences with numbers that do not matter are added.

    python mathworld.py            prints examples
"""
import random
from fractions import Fraction

NAMES = ["tom", "ann", "bob", "mary", "john", "lisa", "paul", "emma", "jack", "sara", "mike", "kate", "sam", "nina", "dan", "amy",
         "ben", "lucy", "max", "rose", "luke", "ella", "adam", "jane", "carl", "nora", "eric", "ruth", "alex", "tina"]
THINGS = ["apples", "books", "marbles", "pencils", "cookies", "stickers", "cards", "balloons", "coins", "flowers", "toys", "shells",
          "stamps", "candies", "eggs", "crayons", "oranges", "bottles", "cupcakes", "erasers", "pens", "peaches", "buttons", "blocks",
          "tickets", "cakes", "bananas", "rocks", "notebooks", "cars", "dolls", "seeds", "fish", "trees", "pages", "pictures"]
BOXES = [("box", "boxes"), ("bag", "bags"), ("basket", "baskets"), ("jar", "jars"), ("crate", "crates"), ("pack", "packs"),
         ("shelf", "shelves"), ("row", "rows"), ("tray", "trays"), ("bucket", "buckets"), ("carton", "cartons"), ("pile", "piles")]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
COLOURS = ["red", "blue", "green", "yellow", "white", "black", "big", "small", "new", "old"]
HAVE = ["has", "owns", "collected", "picked", "bought", "found", "made", "saw", "counted", "packed"]
DID = ["picked", "bought", "found", "made", "sold", "read", "collected", "baked", "painted", "washed"]
GAIN = ["bought", "found", "got", "picked", "received", "collected", "made", "won"]
LOSE = ["ate", "lost", "sold", "gave away", "used", "broke", "threw away", "dropped"]
TOTAL = ["in total", "altogether", "in all", "together"]
LEFT = ["now", "left", "still"]


class N:
    """a number in the text"""

    def __init__(self, v):
        self.v = Fraction(v)


def finish(body, quest, answer, kind):
    words, nums = [], []
    for w in body + quest:
        if isinstance(w, N):
            words.append(f"number{len(nums)}")
            nums.append(w.v)
        else:
            words += w.split()
    nq = sum(len(w.split()) if not isinstance(w, N) else 1 for w in quest)
    return {"words": words, "qmask": [0] * (len(words) - nq) + [1] * nq, "nums": nums, "answer": float(answer),
            "eq": None, "type": kind, "eq_text": ""}


def pick(rng, xs, n=1, but=()):
    xs = [x for x in xs if x not in but]
    return rng.sample(xs, n) if n > 1 else rng.choice(xs)


def num(rng, lo=2, hi=60):
    return rng.randint(lo, hi) if rng.random() < 0.8 else rng.randint(lo, hi * 8)


def owners(rng):
    """several owners of one thing"""
    k = rng.choice([2, 2, 3])
    who, thing, verb = pick(rng, NAMES, k), pick(rng, THINGS), pick(rng, HAVE)
    v = [num(rng) for _ in range(k)]
    body = []
    for i in range(k):
        if i and rng.random() < 0.3:
            body[-1:] = ["and", who[i], verb, N(v[i]), thing, "."]
        else:
            body += [who[i], verb, N(v[i]), thing, "."]
    do = "do" if verb in ("has", "owns") else "did"
    base = {"has": "have", "owns": "own", "collected": "collect", "picked": "pick", "bought": "buy", "found": "find", "made": "make",
            "saw": "see", "counted": "count", "packed": "pack"}[verb]
    kinds = ["all", "more"] + (["pair", "pair", "more3"] if k == 3 else [])
    kind = rng.choice(kinds)
    if kind == "all":
        return body, ["how many", thing, do, "they", base, rng.choice(TOTAL), "?"], sum(v), False
    if kind == "pair":
        a, b = rng.sample(range(k), 2)
        return body, ["how many", thing, do, who[a], "and", who[b], base, rng.choice(TOTAL), "?"], v[a] + v[b], True
    if kind == "more":
        a, b = rng.sample(range(k), 2)
        if v[a] == v[b]:
            return None
        if v[a] < v[b]:
            a, b = b, a
        does = "does" if do == "do" else "did"
        if rng.random() < 0.6:
            return body, ["how many more", thing, does, who[a], base, "than", who[b], "?"], v[a] - v[b], True
        return body, ["how many fewer", thing, does, who[b], base, "than", who[a], "?"], v[a] - v[b], True
    a, b, c = rng.sample(range(k), 3)
    if v[a] + v[b] <= v[c]:
        return None
    return body, ["how many more", thing, do, who[a], "and", who[b], base, "than", who[c], "?"], v[a] + v[b] - v[c], True


def sorts(rng):
    """one owner: on different days, or of different colours"""
    k = rng.choice([2, 3, 3])
    who, thing, verb = pick(rng, NAMES), pick(rng, THINGS), pick(rng, DID)
    v = [num(rng) for _ in range(k)]
    days = rng.random() < 0.5
    tag = pick(rng, DAYS if days else COLOURS, k)
    body = [who, verb]
    for i in range(k):
        if i:
            body.append("and" if i == k - 1 else ("," if rng.random() < 0.5 else "and"))
        body += [N(v[i]), thing, "on", tag[i]] if days else [N(v[i]), tag[i], thing]
    body.append(".")
    base = {"picked": "pick", "bought": "buy", "found": "find", "made": "make", "sold": "sell", "read": "read", "collected": "collect",
            "baked": "bake", "painted": "paint", "washed": "wash"}[verb]
    kind = rng.choice(["all", "more"] + (["pair", "pair", "more"] if k == 3 else []))
    if kind == "all":
        return body, ["how many", thing, "did", who, base, rng.choice(TOTAL), "?"], sum(v), False
    a, b = rng.sample(range(k), 2)
    if kind == "pair":
        if days:
            return body, ["how many", thing, "did", who, base, "on", tag[a], "and", tag[b], "?"], v[a] + v[b], True
        return body, ["how many", tag[a], "and", tag[b], thing, "did", who, base, "?"], v[a] + v[b], True
    if v[a] == v[b]:
        return None
    if v[a] < v[b]:
        a, b = b, a
    if days:
        return body, ["how many more", thing, "did", who, base, "on", tag[a], "than on", tag[b], "?"], v[a] - v[b], True
    return body, ["how many more", tag[a], thing, "than", tag[b], thing, "did", who, base, "?"], v[a] - v[b], True


def change(rng):
    """had - got or lost - has now; any of the three can be the unknown"""
    who, thing = pick(rng, NAMES), pick(rng, THINGS)
    he = rng.choice(["he", "she", who])
    gain = rng.random() < 0.5
    verb = pick(rng, GAIN if gain else LOSE)
    start, d = num(rng, 10, 90), num(rng, 2, 40)
    if not gain and d >= start:
        start, d = start + d, d
    end = start + d if gain else start - d
    more = ["more"] if gain and rng.random() < 0.7 else []
    kind = rng.choice(["end", "end", "start", "change", "two", "two_part"])
    if kind == "end":
        body = [who, "had", N(start), thing, ".", he, verb, N(d)] + more + ([thing] if rng.random() < 0.5 else []) + ["."]
        return body, ["how many", thing, "does", who, "have", rng.choice(LEFT if not gain else ["now", "in total", "in all"]), "?"], end, False
    if kind == "start":
        body = [who, "had some", thing, ".", he, verb, N(d)] + more + ["."] + rng.choice([["now", he, "has", N(end), thing, "."],
                                                                                           [he, "has", N(end), thing, "left ." if not gain else "now ."]])
        return body, ["how many", thing, "did", who, "have", rng.choice(["at first", "in the beginning", "at the start", "before"]), "?"], start, False
    if kind == "change":
        body = [who, "had", N(start), thing, ".", he, verb, "some", "more ." if gain else ".", "now", he, "has", N(end), thing, "."]
        base = {"bought": "buy", "found": "find", "got": "get", "picked": "pick", "received": "receive", "collected": "collect", "made": "make",
                "won": "win", "ate": "eat", "lost": "lose", "sold": "sell", "gave away": "give away", "used": "use", "broke": "break",
                "threw away": "throw away", "dropped": "drop"}[verb]
        return body, ["how many", thing, "did", who, base, "?"], d, False
    d2 = num(rng, 2, 30)
    if not gain and d + d2 >= start:
        start = d + d2 + num(rng, 1, 40)
    body = [who, "had", N(start), thing, ".", he, verb, N(d), thing, ".", "then", he, verb, N(d2), "more", "."]
    if kind == "two":
        end2 = start + d + d2 if gain else start - d - d2
        return body, ["how many", thing, "does", who, "have", "now" if gain else rng.choice(LEFT), "?"], end2, False
    base = verb.split()[0]
    return body, ["how many", thing, "did", who, {"ate": "eat", "lost": "lose", "sold": "sell", "gave": "give away", "used": "use", "broke": "break",
                                                   "threw": "throw away", "dropped": "drop", "bought": "buy", "found": "find", "got": "get",
                                                   "picked": "pick", "received": "receive", "collected": "collect", "made": "make", "won": "win"}[base],
                  rng.choice(["in total", "in all", "altogether"]), "?"], d + d2, False


def compare(rng):
    """more than, fewer than: said about one, asked about the other"""
    a, b = pick(rng, NAMES, 2)
    thing = pick(rng, THINGS)
    x, d = num(rng, 10, 90), num(rng, 2, 9 if rng.random() < 0.5 else 40)
    more = rng.random() < 0.5
    word = "more" if more else rng.choice(["fewer", "less"])
    if rng.random() < 0.5:                      # b is told through a: b = a +- d
        if not more and d >= x:
            x += d
        body = [a, "has", N(x), thing, ".", b, "has", N(d), word, thing, "than", a, "."]
        return body, ["how many", thing, "does", b, "have", "?"], x + d if more else x - d, True
    if more and d >= x:                         # a is told, and that a has d more / fewer than b: b = a -+ d
        x += d
    body = [a, "has", N(x), thing, "."] + rng.choice([[a, "has"], ["that is"], ["he has"], ["she has"]]) + [N(d), word, thing, "than", b, "."]
    return body, ["how many", thing, "does", b, "have", "?"], x - d if more else x + d, True


def groups(rng):
    """equal groups and prices: times and divided, alone and after one more step"""
    who, thing = pick(rng, NAMES), pick(rng, THINGS)
    box, boxes = pick(rng, BOXES)
    g, each = num(rng, 2, 12), num(rng, 2, 15)
    total = g * each
    kind = rng.choice(["times", "times", "each", "count", "price", "unit", "times_minus", "minus_each", "plus_each"])
    if kind == "times":
        body = rng.choice([[who, "has", N(g), boxes, ".", "each", box, "has", N(each), thing, "."],
                           ["there are", N(g), boxes, ".", "there are", N(each), thing, "in each", box, "."],
                           [who, "has", N(g), boxes, "with", N(each), thing, "in each", box, "."],
                           ["each", box, "has", N(each), thing, ".", who, "has", N(g), boxes, "."]])
        return body, rng.choice([["how many", thing, "does", who, "have", rng.choice(["in total", "in all", "altogether"]), "?"],
                                 ["how many", thing, "are there", rng.choice(["in total", "in all", "altogether"]), "?"]]), total, False
    if kind == "each":
        body = [who, "has", N(total), thing, ".", rng.choice(["he", "she", who]), "puts them equally into", N(g), boxes, "."]
        return body, ["how many", thing, rng.choice(["are in each", "go into each", "will be in each"]), box, "?"], each, False
    if kind == "count":
        body = [who, "has", N(total), thing, ".", rng.choice(["he", "she", who]), "puts", N(each), thing, "in each", box, "."]
        return body, ["how many", boxes, "does", who, rng.choice(["need", "use", "fill"]), "?"], g, False
    if kind == "price":
        body = rng.choice([["each of the", thing, "costs", N(each), "dollars", ".", who, "bought", N(g), "of them", "."],
                           [who, "bought", N(g), thing, "for", N(each), "dollars each", "."]])
        return body, rng.choice([["how much did", who, "pay", "?"], ["how much did", who, "spend", rng.choice(["in total", "in all"]), "?"],
                                 ["how many dollars did", who, "spend", "?"]]), total, False
    if kind == "unit":
        body = [who, "paid", N(total), "dollars for", N(g), thing, "."]
        return body, rng.choice([["how much does each of the", thing, "cost", "?"], ["how much did", who, "pay for each", "?"]]), each, False
    if kind == "times_minus":
        d = num(rng, 1, total - 1)
        verb = pick(rng, LOSE)
        body = [who, "has", N(g), boxes, "with", N(each), thing, "in each", box, ".", rng.choice(["he", "she", who]), verb, N(d), thing, "."]
        return body, ["how many", thing, "does", who, "have", rng.choice(LEFT), "?"], total - d, False
    d = num(rng, 2, 40)
    if kind == "minus_each":
        body = [who, "had", N(total + d), thing, ".", rng.choice(["he", "she", who]), pick(rng, LOSE), N(d), "of them", ".",
                "the rest were put equally into", N(g), boxes, "."]
        return body, ["how many", thing, rng.choice(["are in each", "went into each"]), box, "?"], each, False
    if d >= total:
        d = total - 1
    body = [who, "has", N(total - d), pick(rng, COLOURS[:6]), thing, "and", N(d), pick(rng, COLOURS[6:]), thing, ".",
            rng.choice(["he", "she", who]), "puts them equally into", N(g), boxes, "."]
    return body, ["how many", thing, rng.choice(["are in each", "go into each"]), box, "?"], each, False


def sorts_compare(rng):
    """one owner, two days or two colours, one of them told through the other"""
    who, thing, verb = pick(rng, NAMES), pick(rng, THINGS), pick(rng, DID)
    he = rng.choice(["he", "she", who])
    days = rng.random() < 0.6
    a, b = pick(rng, DAYS if days else COLOURS, 2)
    x, d = num(rng, 10, 90), num(rng, 2, 9 if rng.random() < 0.5 else 40)
    more = rng.random() < 0.5
    word = "more" if more else "fewer"
    told = rng.choice([a, b])                       # the sentence is "d more on a than on b"; x is told for one of them
    if (told == a) == more and d >= x:              # the other one is x - d
        x += d
    base = {"picked": "pick", "bought": "buy", "found": "find", "made": "make", "sold": "sell", "read": "read", "collected": "collect",
            "baked": "bake", "painted": "paint", "washed": "wash"}[verb]
    if days:
        body = [who, verb, N(x), thing, "on", told, ".", he, verb, N(d), word, thing, "on", a, "than on", b, "."]
        ask = a if told == b else b
        quest = ["how many", thing, "did", who, base, "on", ask, "?"]
    else:
        body = [who, verb, N(x), told, thing, ".", he, verb, N(d), word, a, thing, "than", b, thing, "."]
        ask = a if told == b else b
        quest = ["how many", ask, thing, "did", who, base, "?"]
    if told == b:                                   # a = b +- d
        return body, quest, x + d if more else x - d, True
    return body, quest, x - d if more else x + d, True


def rate2(rng):
    """groups with two sorts in each: per group, in all, one sort in all"""
    who, thing = pick(rng, NAMES), pick(rng, THINGS)
    box, boxes = pick(rng, BOXES)
    c1, c2 = pick(rng, COLOURS, 2)
    g, a, b = num(rng, 2, 12), num(rng, 2, 20), num(rng, 2, 20)
    first = rng.choice([["there are", N(g), boxes, "."], [who, "has", N(g), boxes, "."]])
    second = ["each", box, "has", N(a), c1, thing, "and", N(b), c2, thing, "."]
    body = first + second if rng.random() < 0.6 else second + first
    kind = rng.choice(["each", "all", "one", "one", "more"])
    if kind == "each":
        return body, ["how many", thing, "are in each", box, "?"], a + b, True
    if kind == "all":
        return body, ["how many", thing, "are in the", boxes, rng.choice(["altogether", "in total", "in all"]), "?"], (a + b) * g, True
    if kind == "one":
        return body, ["how many", c1, thing, "are in the", boxes, rng.choice(["altogether", "in total", "in all"]), "?"], a * g, True
    if a == b:
        return None
    hi, lo = (c1, c2) if a > b else (c2, c1)
    return body, ["how many more", hi, thing, "than", lo, thing, "are in each", box, "?"], abs(a - b), True


def daily(rng):
    """every day so many: in several days, per day, and back"""
    who, thing, verb = pick(rng, NAMES), pick(rng, THINGS), pick(rng, DID)
    base = {"picked": "picks", "bought": "buys", "found": "finds", "made": "makes", "sold": "sells", "read": "reads", "collected": "collects",
            "baked": "bakes", "painted": "paints", "washed": "washes"}[verb]
    stem = base[:-1] if base != "washes" else "wash"
    a, b, d = num(rng, 2, 30), num(rng, 2, 30), num(rng, 2, 12)
    two = rng.random() < 0.5
    kind = rng.choice(["days", "days", "per day", "part", "back"] if two else ["days", "days", "back"])
    if kind == "back":
        body = [who, verb, N(a * d), thing, "in", N(d), "days", ".", rng.choice(["he", "she", who]), verb, "the same number of", thing, "each day", "."]
        return body, ["how many", thing, "did", who, {"picks": "pick", "buys": "buy", "finds": "find", "makes": "make", "sells": "sell", "reads": "read",
                                                       "collects": "collect", "bakes": "bake", "paints": "paint", "washes": "wash"}[base], "each day", "?"], a, False
    every = rng.choice(["every day", "each day"])
    body = [every, who, base, N(a), thing] + (["in the morning and", N(b), thing, "in the evening"] if two else []) + ["."]
    if rng.random() < 0.5:
        body += [rng.choice(["he", "she", who]), "does this for", N(d), "days", "."]
        tail = [rng.choice(["in all", "in total", "altogether"])]
    else:
        tail = ["in", N(d), "days"]
    if kind == "days":
        return body, ["how many", thing, "does", who, stem] + tail + ["?"], (a + b if two else a) * d, True
    if kind == "per day":
        if isinstance(tail[-1], str) and tail[-1] == "days":
            return None
        return body, ["how many", thing, "does", who, stem, "each day", "?"], a + b, True
    return body, ["how many", thing, "does", who, stem, "in the morning"] + tail + ["?"], a * d, True


def owners_change(rng):
    """two owners, then one of them gets or loses some"""
    a, b = pick(rng, NAMES, 2)
    thing = pick(rng, THINGS)
    x, y, d = num(rng, 5, 60), num(rng, 5, 60), num(rng, 2, 30)
    gain = rng.random() < 0.6
    if not gain and d >= x:
        x += d
    verb = pick(rng, GAIN if gain else LOSE)
    body = rng.choice([[a, "had", N(x), thing, "and", b, "had", N(y), thing, "."], [a, "had", N(x), thing, ".", b, "had", N(y), thing, "."]])
    body += [a, "then", verb, N(d)] + (["more"] if gain else []) + [thing, "."]
    now = x + d if gain else x - d
    kind = rng.choice(["both", "one", "more", "other"])
    if kind == "both":
        return body, ["how many", thing, "do", a, "and", b, "have", rng.choice(["now", "in total", "altogether", "together"]), "?"], now + y, True
    if kind == "one":
        return body, ["how many", thing, "does", a, "have", "now", "?"], now, True
    if kind == "other":
        return None if rng.random() < 0.7 else (body + [], ["how many", thing, "do", a, "and", b, "have", "in all", "?"], now + y, True)
    if now == y:
        return None
    hi, lo = (a, b) if now > y else (b, a)
    return body, ["how many more", thing, "does", hi, "have than", lo, "now", "?"], abs(now - y), True


def part(rng):
    """a whole and its parts: the rest, what was given away"""
    who, thing = pick(rng, NAMES), pick(rng, THINGS)
    c1, c2 = pick(rng, COLOURS, 2)
    a, b = num(rng, 2, 60), num(rng, 2, 60)
    kind = rng.choice(["rest", "rest", "gave left", "gave all", "whole"])
    if kind == "rest":
        body = [who, "has", N(a + b), thing, ".", N(a), "of them are", c1, "and the rest are", c2, "."]
        return body, ["how many", c2, thing, "does", who, "have", "?"], b, True
    if kind == "whole":
        body = [who, "has", N(a), c1, thing, "and", N(b), c2, thing, "."]
        return body, ["how many", thing, "does", who, "have", rng.choice(TOTAL[:3]), "?"], a + b, False
    p1, p2 = pick(rng, NAMES, 2, but=(who,))
    rest = num(rng, 1, 40)
    body = [who, "had", N(a + b + rest), thing, ".", rng.choice(["he", "she", who]), "gave", N(a), "to", p1, "and", N(b), "to", p2, "."]
    if kind == "gave left":
        return body, ["how many", thing, "does", who, "have", "left", "?"], rest, False
    return body, ["how many", thing, "did", who, "give away", rng.choice(["in total", "in all", "altogether"]), "?"], a + b, False


def noise(rng, body, quest, named):
    """a sentence with a number that does not matter for the question"""
    used = {w for w in body + quest if isinstance(w, str)}
    who = [w for w in body if isinstance(w, str) and w in NAMES]
    thing = [w for w in body if isinstance(w, str) and w in THINGS]
    kinds = ["age", "other thing"] + (["other owner"] if named and thing else [])
    kind = rng.choice(kinds)
    if kind == "age" or not who:
        s = rng.choice([[pick(rng, who or NAMES), "is", N(num(rng, 6, 15)), "years old", "."],
                        ["it took", N(num(rng, 2, 50)), "minutes", "."],
                        ["the shop is", N(num(rng, 2, 30)), "miles away", "."]])
    elif kind == "other thing":
        s = [pick(rng, who), "also has", N(num(rng)), pick(rng, THINGS, but=used), "."]
    else:
        s = [pick(rng, NAMES, but=used), "has", N(num(rng)), thing[0], "."]
    ends = [i + 1 for i, w in enumerate(body) if w == "."]
    at = rng.choice([0] + [e for e in ends if e < len(body) or body[-1] == "."])
    return body[:at] + s + body[at:]


KINDS = [owners, sorts, change, compare, groups, sorts_compare, rate2, daily, owners_change, part]


def problem(rng):
    while True:
        f = rng.choice(KINDS)
        got = f(rng)
        if not got:
            continue
        body, quest, ans, named = got
        if ans <= 0:
            continue
        ends = [i for i, w in enumerate(body) if w == "."]
        if len(ends) >= 2 and rng.random() < 0.25:           # "... . if <the last sentence> how many ..."
            body = body[:ends[-2] + 1] + ["if"] + body[ends[-2] + 1:-1] + ([","] if rng.random() < 0.4 else [])
        r = rng.random()
        if r < 0.45:
            body = noise(rng, body, quest, named)
            if r < 0.12:
                body = noise(rng, body, quest, named)
        return finish(body, quest, ans, f.__name__)


if __name__ == "__main__":
    rng = random.Random(1)
    for _ in range(16):
        p = problem(rng)
        print(" ".join(p["words"]), "|", [float(x) for x in p["nums"]], "->", p["answer"])
