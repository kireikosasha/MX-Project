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

import ledger

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
    """a number in the text with its role in the ledger (ledger.py); "0": it does not matter"""

    def __init__(self, v, role="0"):
        self.v = Fraction(v)
        self.role = role


def finish(body, quest, answer, kind, query="val"):
    words, nums, roles = [], [], []
    for w in body + quest:
        if isinstance(w, N):
            words.append(f"number{len(nums)}")
            nums.append(w.v)
            roles.append(ledger.R[w.role])
        else:
            words += w.split()
    nq = sum(len(w.split()) if not isinstance(w, N) else 1 for w in quest)
    return {"words": words, "qmask": [0] * (len(words) - nq) + [1] * nq, "nums": nums, "answer": float(answer),
            "eq": None, "type": kind, "eq_text": "", "roles": roles, "query": ledger.Q[query]}


def pick(rng, xs, n=1, but=()):
    xs = [x for x in xs if x not in but]
    return rng.sample(xs, n) if n > 1 else rng.choice(xs)


def num(rng, lo=2, hi=60):
    return rng.randint(lo, hi) if rng.random() < 0.8 else rng.randint(lo, hi * 8)


def roles(n, which, role):
    for i in which:
        n[i].role = role


def two_sides(n, one, other):
    """two accounts that are compared: A is the one whose number comes first in the text"""
    first = one if min(one) < min(other) else other
    roles(n, first, "+A")
    roles(n, other if first is one else one, "+B")


def owners(rng):
    """several owners of one thing"""
    k = rng.choice([2, 2, 3])
    who, thing, verb = pick(rng, NAMES, k), pick(rng, THINGS), pick(rng, HAVE)
    v = [num(rng) for _ in range(k)]
    n = [N(x) for x in v]
    body = []
    for i in range(k):
        if i and rng.random() < 0.3:
            body[-1:] = ["and", who[i], verb, n[i], thing, "."]
        else:
            body += [who[i], verb, n[i], thing, "."]
    do = "do" if verb in ("has", "owns") else "did"
    base = {"has": "have", "owns": "own", "collected": "collect", "picked": "pick", "bought": "buy", "found": "find", "made": "make",
            "saw": "see", "counted": "count", "packed": "pack"}[verb]
    kinds = ["all", "more"] + (["pair", "pair", "more3"] if k == 3 else [])
    kind = rng.choice(kinds)
    if kind == "all":
        roles(n, range(k), "+A")
        return body, ["how many", thing, do, "they", base, rng.choice(TOTAL), "?"], sum(v), False, "val"
    if kind == "pair":
        a, b = rng.sample(range(k), 2)
        roles(n, (a, b), "+A")
        return body, ["how many", thing, do, who[a], "and", who[b], base, rng.choice(TOTAL), "?"], v[a] + v[b], True, "val"
    if kind == "more":
        a, b = rng.sample(range(k), 2)
        if v[a] == v[b]:
            return None
        if v[a] < v[b]:
            a, b = b, a
        does = "does" if do == "do" else "did"
        two_sides(n, (a,), (b,))
        if rng.random() < 0.6:
            return body, ["how many more", thing, does, who[a], base, "than", who[b], "?"], v[a] - v[b], True, "dif"
        return body, ["how many fewer", thing, does, who[b], base, "than", who[a], "?"], v[a] - v[b], True, "dif"
    a, b, c = rng.sample(range(k), 3)
    if v[a] + v[b] <= v[c]:
        return None
    two_sides(n, (a, b), (c,))
    return body, ["how many more", thing, do, who[a], "and", who[b], base, "than", who[c], "?"], v[a] + v[b] - v[c], True, "dif"


def sorts(rng):
    """one owner: on different days, or of different colours"""
    k = rng.choice([2, 3, 3])
    who, thing, verb = pick(rng, NAMES), pick(rng, THINGS), pick(rng, DID)
    v = [num(rng) for _ in range(k)]
    days = rng.random() < 0.5
    tag = pick(rng, DAYS if days else COLOURS, k)
    n = [N(x) for x in v]
    body = [who, verb]
    for i in range(k):
        if i:
            body.append("and" if i == k - 1 else ("," if rng.random() < 0.5 else "and"))
        body += [n[i], thing, "on", tag[i]] if days else [n[i], tag[i], thing]
    body.append(".")
    base = {"picked": "pick", "bought": "buy", "found": "find", "made": "make", "sold": "sell", "read": "read", "collected": "collect",
            "baked": "bake", "painted": "paint", "washed": "wash"}[verb]
    kind = rng.choice(["all", "more"] + (["pair", "pair", "more"] if k == 3 else []))
    if kind == "all":
        roles(n, range(k), "+A")
        return body, ["how many", thing, "did", who, base, rng.choice(TOTAL), "?"], sum(v), False, "val"
    a, b = rng.sample(range(k), 2)
    if kind == "pair":
        roles(n, (a, b), "+A")
        if days:
            return body, ["how many", thing, "did", who, base, "on", tag[a], "and", tag[b], "?"], v[a] + v[b], True, "val"
        return body, ["how many", tag[a], "and", tag[b], thing, "did", who, base, "?"], v[a] + v[b], True, "val"
    if v[a] == v[b]:
        return None
    if v[a] < v[b]:
        a, b = b, a
    two_sides(n, (a,), (b,))
    if days:
        return body, ["how many more", thing, "did", who, base, "on", tag[a], "than on", tag[b], "?"], v[a] - v[b], True, "dif"
    return body, ["how many more", tag[a], thing, "than", tag[b], thing, "did", who, base, "?"], v[a] - v[b], True, "dif"


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
    sign = "+A" if gain else "-A"
    if kind == "end":
        body = [who, "had", N(start, "+A"), thing, ".", he, verb, N(d, sign)] + more + ([thing] if rng.random() < 0.5 else []) + ["."]
        return body, ["how many", thing, "does", who, "have", rng.choice(LEFT if not gain else ["now", "in total", "in all"]), "?"], end, False, "val"
    if kind == "start":
        last = N(end, "=A")
        body = [who, "had some", thing, ".", he, verb, N(d, sign)] + more + ["."] + rng.choice([["now", he, "has", last, thing, "."],
                                                                                                 [he, "has", last, thing, "left ." if not gain else "now ."]])
        return body, ["how many", thing, "did", who, "have", rng.choice(["at first", "in the beginning", "at the start", "before"]), "?"], start, False, "mis"
    if kind == "change":
        body = [who, "had", N(start, "+A"), thing, ".", he, verb, "some", "more ." if gain else ".", "now", he, "has", N(end, "=A"), thing, "."]
        base = {"bought": "buy", "found": "find", "got": "get", "picked": "pick", "received": "receive", "collected": "collect", "made": "make",
                "won": "win", "ate": "eat", "lost": "lose", "sold": "sell", "gave away": "give away", "used": "use", "broke": "break",
                "threw away": "throw away", "dropped": "drop"}[verb]
        return body, ["how many", thing, "did", who, base, "?"], d, False, "mis"
    d2 = num(rng, 2, 30)
    if not gain and d + d2 >= start:
        start = d + d2 + num(rng, 1, 40)
    n0 = N(start, "+A" if kind == "two" else "0")
    body = [who, "had", n0, thing, ".", he, verb, N(d, sign if kind == "two" else "+A"), thing, ".",
            "then", he, verb, N(d2, sign if kind == "two" else "+A"), "more", "."]
    if kind == "two":
        end2 = start + d + d2 if gain else start - d - d2
        return body, ["how many", thing, "does", who, "have", "now" if gain else rng.choice(LEFT), "?"], end2, False, "val"
    base = verb.split()[0]
    return body, ["how many", thing, "did", who, {"ate": "eat", "lost": "lose", "sold": "sell", "gave": "give away", "used": "use", "broke": "break",
                                                   "threw": "throw away", "dropped": "drop", "bought": "buy", "found": "find", "got": "get",
                                                   "picked": "pick", "received": "receive", "collected": "collect", "made": "make", "won": "win"}[base],
                  rng.choice(["in total", "in all", "altogether"]), "?"], d + d2, False, "val"


def compare(rng):
    """more than, fewer than: said about one, asked about the other"""
    a, b = pick(rng, NAMES, 2)
    thing = pick(rng, THINGS)
    x, d = num(rng, 10, 90), num(rng, 2, 9 if rng.random() < 0.5 else 40)
    more = rng.random() < 0.5
    word = "more" if more else rng.choice(["fewer", "less"])
    if rng.random() < 0.5:                      # b is told through a: b = a +- d. The asked one (A) is b
        if not more and d >= x:
            x += d
        body = [a, "has", N(x, "+B"), thing, ".", b, "has", N(d, ">" if more else "<"), word, thing, "than", a, "."]
        return body, ["how many", thing, "does", b, "have", "?"], x + d if more else x - d, True, "val"
    if more and d >= x:                         # a is told, and that a has d more / fewer than b: b = a -+ d
        x += d
    body = [a, "has", N(x, "+B"), thing, "."] + rng.choice([[a, "has"], ["that is"], ["he has"], ["she has"]]) + [N(d, "<" if more else ">"), word, thing, "than", b, "."]
    return body, ["how many", thing, "does", b, "have", "?"], x - d if more else x + d, True, "val"


def groups(rng):
    """equal groups and prices: times and divided, alone and after one more step"""
    who, thing = pick(rng, NAMES), pick(rng, THINGS)
    box, boxes = pick(rng, BOXES)
    g, each = num(rng, 2, 12), num(rng, 2, 15)
    total = g * each
    kind = rng.choice(["times", "times", "each", "count", "price", "unit", "times_minus", "minus_each", "plus_each"])
    ng, ne = N(g, "*A"), N(each, "+A")          # so many in each: the amount; so many groups: its multiplier
    if kind == "times":
        body = rng.choice([[who, "has", ng, boxes, ".", "each", box, "has", ne, thing, "."],
                           ["there are", ng, boxes, ".", "there are", ne, thing, "in each", box, "."],
                           [who, "has", ng, boxes, "with", ne, thing, "in each", box, "."],
                           ["each", box, "has", ne, thing, ".", who, "has", ng, boxes, "."]])
        return body, rng.choice([["how many", thing, "does", who, "have", rng.choice(["in total", "in all", "altogether"]), "?"],
                                 ["how many", thing, "are there", rng.choice(["in total", "in all", "altogether"]), "?"]]), total, False, "val"
    if kind == "each":
        body = [who, "has", N(total, "+A"), thing, ".", rng.choice(["he", "she", who]), "puts them equally into", N(g, "/A"), boxes, "."]
        return body, ["how many", thing, rng.choice(["are in each", "go into each", "will be in each"]), box, "?"], each, False, "val"
    if kind == "count":
        body = [who, "has", N(total, "+A"), thing, ".", rng.choice(["he", "she", who]), "puts", N(each, "/A"), thing, "in each", box, "."]
        return body, ["how many", boxes, "does", who, rng.choice(["need", "use", "fill"]), "?"], g, False, "val"
    if kind == "price":
        body = rng.choice([["each of the", thing, "costs", ne, "dollars", ".", who, "bought", ng, "of them", "."],
                           [who, "bought", ng, thing, "for", ne, "dollars each", "."]])
        return body, rng.choice([["how much did", who, "pay", "?"], ["how much did", who, "spend", rng.choice(["in total", "in all"]), "?"],
                                 ["how many dollars did", who, "spend", "?"]]), total, False, "val"
    if kind == "unit":
        body = [who, "paid", N(total, "+A"), "dollars for", N(g, "/A"), thing, "."]
        return body, rng.choice([["how much does each of the", thing, "cost", "?"], ["how much did", who, "pay for each", "?"]]), each, False, "val"
    if kind == "times_minus":                   # two accounts: what was in the boxes, and what is gone
        d = num(rng, 1, total - 1)
        verb = pick(rng, LOSE)
        body = [who, "has", ng, boxes, "with", ne, thing, "in each", box, ".", rng.choice(["he", "she", who]), verb, N(d, "+B"), thing, "."]
        return body, ["how many", thing, "does", who, "have", rng.choice(LEFT), "?"], total - d, False, "dif"
    d = num(rng, 2, 40)
    if kind == "minus_each":
        body = [who, "had", N(total + d, "+A"), thing, ".", rng.choice(["he", "she", who]), pick(rng, LOSE), N(d, "-A"), "of them", ".",
                "the rest were put equally into", N(g, "/A"), boxes, "."]
        return body, ["how many", thing, rng.choice(["are in each", "went into each"]), box, "?"], each, False, "val"
    if d >= total:
        d = total - 1
    body = [who, "has", N(total - d, "+A"), pick(rng, COLOURS[:6]), thing, "and", N(d, "+A"), pick(rng, COLOURS[6:]), thing, ".",
            rng.choice(["he", "she", who]), "puts them equally into", N(g, "/A"), boxes, "."]
    return body, ["how many", thing, rng.choice(["are in each", "go into each"]), box, "?"], each, False, "val"


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
    nx = N(x, "+B")                                 # the asked one (A) is told through the other one
    nd = N(d, (">" if more else "<") if told == b else ("<" if more else ">"))
    if days:
        body = [who, verb, nx, thing, "on", told, ".", he, verb, nd, word, thing, "on", a, "than on", b, "."]
        ask = a if told == b else b
        quest = ["how many", thing, "did", who, base, "on", ask, "?"]
    else:
        body = [who, verb, nx, told, thing, ".", he, verb, nd, word, a, thing, "than", b, thing, "."]
        ask = a if told == b else b
        quest = ["how many", ask, thing, "did", who, base, "?"]
    if told == b:                                   # a = b +- d
        return body, quest, x + d if more else x - d, True, "val"
    return body, quest, x - d if more else x + d, True, "val"


def rate2(rng):
    """groups with two sorts in each: per group, in all, one sort in all"""
    who, thing = pick(rng, NAMES), pick(rng, THINGS)
    box, boxes = pick(rng, BOXES)
    c1, c2 = pick(rng, COLOURS, 2)
    g, a, b = num(rng, 2, 12), num(rng, 2, 20), num(rng, 2, 20)
    kind = rng.choice(["each", "all", "one", "one", "more"])
    ng = N(g, "*A" if kind in ("all", "one") else "0")
    na, nb = N(a, "+A"), N(b, {"each": "+A", "all": "+A", "one": "0", "more": "+B"}[kind])
    first = rng.choice([["there are", ng, boxes, "."], [who, "has", ng, boxes, "."]])
    second = ["each", box, "has", na, c1, thing, "and", nb, c2, thing, "."]
    body = first + second if rng.random() < 0.6 else second + first
    if kind == "each":
        return body, ["how many", thing, "are in each", box, "?"], a + b, True, "val"
    if kind == "all":
        return body, ["how many", thing, "are in the", boxes, rng.choice(["altogether", "in total", "in all"]), "?"], (a + b) * g, True, "val"
    if kind == "one":
        return body, ["how many", c1, thing, "are in the", boxes, rng.choice(["altogether", "in total", "in all"]), "?"], a * g, True, "val"
    if a == b:
        return None
    hi, lo = (c1, c2) if a > b else (c2, c1)
    return body, ["how many more", hi, thing, "than", lo, thing, "are in each", box, "?"], abs(a - b), True, "dif"


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
        body = [who, verb, N(a * d, "+A"), thing, "in", N(d, "/A"), "days", ".", rng.choice(["he", "she", who]), verb, "the same number of", thing, "each day", "."]
        return body, ["how many", thing, "did", who, {"picks": "pick", "buys": "buy", "finds": "find", "makes": "make", "sells": "sell", "reads": "read",
                                                       "collects": "collect", "bakes": "bake", "paints": "paint", "washes": "wash"}[base], "each day", "?"], a, False, "val"
    every = rng.choice(["every day", "each day"])
    nd = N(d, "0" if kind == "per day" else "*A")
    body = [every, who, base, N(a, "+A"), thing] + (["in the morning and", N(b, "0" if kind == "part" else "+A"), thing, "in the evening"] if two else []) + ["."]
    if rng.random() < 0.5:
        body += [rng.choice(["he", "she", who]), "does this for", nd, "days", "."]
        tail = [rng.choice(["in all", "in total", "altogether"])]
    else:
        tail = ["in", nd, "days"]
    if kind == "days":
        return body, ["how many", thing, "does", who, stem] + tail + ["?"], (a + b if two else a) * d, True, "val"
    if kind == "per day":
        if isinstance(tail[-1], str) and tail[-1] == "days":
            return None
        return body, ["how many", thing, "does", who, stem, "each day", "?"], a + b, True, "val"
    return body, ["how many", thing, "does", who, stem, "in the morning"] + tail + ["?"], a * d, True, "val"


def owners_change(rng):
    """two owners, then one of them gets or loses some"""
    a, b = pick(rng, NAMES, 2)
    thing = pick(rng, THINGS)
    x, y, d = num(rng, 5, 60), num(rng, 5, 60), num(rng, 2, 30)
    gain = rng.random() < 0.6
    if not gain and d >= x:
        x += d
    verb = pick(rng, GAIN if gain else LOSE)
    kind = rng.choice(["both", "both", "one", "one", "more", "more"])
    nx, ny, nd = N(x, "+A"), N(y, {"both": "+A", "one": "0", "more": "+B"}[kind]), N(d, "+A" if gain else "-A")
    body = rng.choice([[a, "had", nx, thing, "and", b, "had", ny, thing, "."], [a, "had", nx, thing, ".", b, "had", ny, thing, "."]])
    body += [a, "then", verb, nd] + (["more"] if gain else []) + [thing, "."]
    now = x + d if gain else x - d
    if kind == "both":
        return body, ["how many", thing, "do", a, "and", b, "have", rng.choice(["now", "in total", "altogether", "together", "in all"]), "?"], now + y, True, "val"
    if kind == "one":
        return body, ["how many", thing, "does", a, "have", "now", "?"], now, True, "val"
    if now == y:
        return None
    hi, lo = (a, b) if now > y else (b, a)
    return body, ["how many more", thing, "does", hi, "have than", lo, "now", "?"], abs(now - y), True, "dif"


def part(rng):
    """a whole and its parts: the rest, what was given away"""
    who, thing = pick(rng, NAMES), pick(rng, THINGS)
    c1, c2 = pick(rng, COLOURS, 2)
    a, b = num(rng, 2, 60), num(rng, 2, 60)
    kind = rng.choice(["rest", "rest", "gave left", "gave all", "whole"])
    if kind == "rest":                              # the whole and one part are told, the other part is asked
        body = [who, "has", N(a + b, "=S"), thing, ".", N(a, "+B"), "of them are", c1, "and the rest are", c2, "."]
        return body, ["how many", c2, thing, "does", who, "have", "?"], b, True, "val"
    if kind == "whole":
        body = [who, "has", N(a, "+A"), c1, thing, "and", N(b, "+A"), c2, thing, "."]
        return body, ["how many", thing, "does", who, "have", rng.choice(TOTAL[:3]), "?"], a + b, False, "val"
    p1, p2 = pick(rng, NAMES, 2, but=(who,))
    rest = num(rng, 1, 40)
    left = kind == "gave left"
    body = [who, "had", N(a + b + rest, "+A" if left else "0"), thing, ".", rng.choice(["he", "she", who]), "gave",
            N(a, "-A" if left else "+A"), "to", p1, "and", N(b, "-A" if left else "+A"), "to", p2, "."]
    if left:
        return body, ["how many", thing, "does", who, "have", "left", "?"], rest, False, "val"
    return body, ["how many", thing, "did", who, "give away", rng.choice(["in total", "in all", "altogether"]), "?"], a + b, False, "val"


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
        body, quest, ans, named, query = got
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
        return finish(body, quest, ans, f.__name__, query)


if __name__ == "__main__":
    rng = random.Random(1)
    for _ in range(16):
        p = problem(rng)
        print(" ".join(p["words"]), "|", [float(x) for x in p["nums"]], "->", p["answer"])
