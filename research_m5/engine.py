#!/usr/bin/env python3
"""
The exact part: a store of facts, rules over it, and queries.

The network (translator.py) only turns a sentence into a frame - a predicate and its arguments, which are
words of the sentence itself. Everything that is logic happens here, in code: where things are, who carries
what, order in time, either / or, negation, deduction from kinds, induction from examples, transitive
relations, positions, shortest paths. Nothing here is learned, so it cannot be "almost right".

A frame is (predicate, {role: [values]}); roles: A1 A2 A3 are entities (word spans), D a direction word,
T a time word.
"""
from collections import deque

STATEMENTS = ["MOVE", "NOT_AT", "EITHER", "GET", "DROP", "GIVE", "REL", "ISA", "AFRAID", "PROP", "FITS", "BIGGER"]
QUERIES = ["Q_WHERE", "Q_WHERE_BEFORE", "Q_IS_AT", "Q_COUNT", "Q_LIST", "Q_GIVE_OBJ", "Q_GIVE_FROM", "Q_GIVE_TO",
           "Q_REL_A1", "Q_REL_A2", "Q_REL_YN", "Q_PATH", "Q_AFRAID", "Q_COLOR", "Q_FITS", "Q_BIGGER", "Q_WHY", "Q_WILL"]
PREDICATES = STATEMENTS + QUERIES
ROLES = ["O", "A1", "A2", "A3", "D", "T"]
# roles a predicate needs (all must be present) and may have
NEEDS = {
    "MOVE": ("A1", "A2"), "NOT_AT": ("A1", "A2"), "EITHER": ("A1", "A2", "A3"), "GET": ("A1", "A2"), "DROP": ("A1", "A2"),
    "GIVE": ("A1", "A2", "A3"), "REL": ("A1", "A2", "D"), "ISA": ("A1", "A2"), "AFRAID": ("A1", "A2"), "PROP": ("A1", "A2"),
    "FITS": ("A1", "A2"), "BIGGER": ("A1", "A2"),
    "Q_WHERE": ("A1",), "Q_WHERE_BEFORE": ("A1", "A2"), "Q_IS_AT": ("A1", "A2"), "Q_COUNT": ("A1",), "Q_LIST": ("A1",),
    "Q_GIVE_OBJ": ("A1", "A3"), "Q_GIVE_FROM": ("A2",), "Q_GIVE_TO": ("A2",), "Q_REL_A1": ("A2", "D"), "Q_REL_A2": ("A1", "D"),
    "Q_REL_YN": ("A1", "A2", "D"), "Q_PATH": ("A1", "A2"), "Q_AFRAID": ("A1",), "Q_COLOR": ("A1",), "Q_FITS": ("A1", "A2"),
    "Q_BIGGER": ("A1", "A2"), "Q_WHY": ("A1",), "Q_WILL": ("A1",),
}
MAY = {"MOVE": ("T",), "Q_GIVE_FROM": ("A3",), "Q_GIVE_TO": ("A1",), "Q_WHY": ("A2",)}

PRONOUNS = {"he", "she", "it", "they"}
TIMES = {"yesterday": 0, "morning": 1, "afternoon": 2, "evening": 3}
DIRS = {"north": "n", "south": "s", "east": "e", "west": "w", "above": "n", "below": "s", "right": "e", "left": "w"}
INV = {"n": "s", "s": "n", "e": "w", "w": "e"}
STEP = {"n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0)}
COUNT = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
IRREGULAR = {"mice": "mouse", "wolves": "wolf", "sheep": "sheep", "geese": "goose", "men": "man", "women": "woman", "children": "child"}
# common sense that is not in the text (the "external base"): where a feeling sends you. Used only when
# the story itself gives no example of it.
KB_GOES = {"hungry": "kitchen", "thirsty": "kitchen", "tired": "bedroom", "bored": "garden"}


def singular(w):
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w.endswith("ves"):
        return w[:-3] + "f"
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


class World:
    def __init__(self):
        self.n = 0                  # statements so far
        self.at = {}                # person -> place or None
        self.maybe = {}             # person -> set of places (either / or)
        self.moves = {}             # person -> [(time rank or None, order, place)]
        self.holder = {}            # object -> person or None
        self.lying = {}             # object -> place it was dropped at
        self.track = {}             # object -> places it has been, in order
        self.carry = {}             # person -> objects in the order taken
        self.gives = []             # (giver, object, receiver)
        self.rel = []               # (a, direction, b): a is <direction> of b
        self.kind = {}              # individual -> kind
        self.kind_at = {}           # individual -> when its kind was told
        self.afraid = {}            # kind -> kind
        self.prop = {}              # entity -> [(order, value)]
        self.bigger = set()         # (a, b): a is bigger than b
        self.last = []              # who "he / she / they" is

    # ---- helpers
    def who(self, names):
        if any(x in PRONOUNS for x in names):
            return list(self.last)
        self.last = list(names)
        return list(names)

    def _seen(self, obj, place):
        if place is not None:
            t = self.track.setdefault(obj, [])
            if not t or t[-1] != place:
                t.append(place)

    def _closure(self):
        big = set(self.bigger)
        grew = True
        while grew:
            grew = False
            for a, b in list(big):
                for c, d in list(big):
                    if b == c and (a, d) not in big:
                        big.add((a, d))
                        grew = True
        return big

    def _edges(self):
        e = {}
        for a, d, b in self.rel:                # from b, going d, you reach a
            e.setdefault(b, []).append((d, a))
            e.setdefault(a, []).append((INV[d], b))
        return e

    # ---- statements
    def tell(self, pred, a):
        self.n += 1
        g = lambda r: a.get(r, [])              # noqa: E731
        if pred == "MOVE":
            rank = TIMES.get(g("T")[0]) if g("T") else None
            for p in self.who(g("A1")):
                self.at[p] = g("A2")[0]
                self.maybe.pop(p, None)
                self.moves.setdefault(p, []).append((rank, self.n, g("A2")[0]))
                for o in self.carry.get(p, []):
                    self._seen(o, self.at[p])
        elif pred == "NOT_AT":
            for p in self.who(g("A1")):
                self.at[p] = None
                self.maybe[p] = set()
        elif pred == "EITHER":
            for p in self.who(g("A1")):
                self.at[p] = None
                self.maybe[p] = {g("A2")[0], g("A3")[0]}
        elif pred == "GET":
            for p in self.who(g("A1")):
                for o in g("A2"):
                    self.holder[o] = p
                    if o not in self.carry.setdefault(p, []):
                        self.carry[p].append(o)
                    self._seen(o, self.at.get(p))
        elif pred == "DROP":
            for p in self.who(g("A1")):
                for o in g("A2"):
                    self.holder[o] = None
                    self.lying[o] = self.at.get(p)
                    if o in self.carry.get(p, []):
                        self.carry[p].remove(o)
        elif pred == "GIVE":
            p, q = self.who(g("A1"))[0], g("A3")[0]
            for o in g("A2"):
                self.holder[o] = q
                if o in self.carry.get(p, []):
                    self.carry[p].remove(o)
                if o not in self.carry.setdefault(q, []):
                    self.carry[q].append(o)
                self.gives.append((p, o, q))
                self._seen(o, self.at.get(q))
        elif pred == "REL":
            self.rel.append((g("A1")[0], DIRS[g("D")[0]], g("A2")[0]))
        elif pred == "ISA":
            self.kind[g("A1")[0]] = singular(g("A2")[0])
            self.kind_at[g("A1")[0]] = self.n
        elif pred == "AFRAID":
            self.afraid[singular(g("A1")[0])] = singular(g("A2")[0])
        elif pred == "PROP":
            self.prop.setdefault(g("A1")[0], []).append((self.n, g("A2")[0]))
        elif pred == "FITS":
            self.bigger.add((g("A2")[0], g("A1")[0]))
        elif pred == "BIGGER":
            self.bigger.add((g("A1")[0], g("A2")[0]))
        else:
            raise ValueError(pred)

    # ---- queries
    def where(self, x):
        if x in self.holder or x in self.lying:
            h = self.holder.get(x)
            return self.at.get(h) if h else self.lying.get(x)
        return self.at.get(x)

    def ask(self, pred, a):
        g = lambda r: a.get(r, [])              # noqa: E731
        x = g("A1")[0] if g("A1") else None
        if pred == "Q_WHERE":
            return self.where(x)
        if pred == "Q_WHERE_BEFORE":
            place = g("A2")[0]
            if x in self.track:
                seq = self.track[x]
            else:
                mv = self.moves.get(x, [])
                timed = all(m[0] is not None for m in mv)
                seq = [m[2] for m in (sorted(mv, key=lambda m: (m[0], m[1])) if timed else mv)]
            for i in range(len(seq) - 1, 0, -1):
                if seq[i] == place:
                    return seq[i - 1]
            return None
        if pred == "Q_IS_AT":
            place = g("A2")[0]
            if self.at.get(x) is not None:
                return "yes" if self.at[x] == place else "no"
            return "maybe" if place in self.maybe.get(x, ()) else "no"
        if pred == "Q_COUNT":
            return COUNT[min(len(self.carry.get(x, [])), len(COUNT) - 1)]
        if pred == "Q_LIST":
            return ",".join(self.carry.get(x, [])) or "nothing"
        if pred in ("Q_GIVE_OBJ", "Q_GIVE_FROM", "Q_GIVE_TO"):
            want = {"A1": 0, "A2": 1, "A3": 2}
            out = {"Q_GIVE_OBJ": 1, "Q_GIVE_FROM": 0, "Q_GIVE_TO": 2}[pred]
            for ev in reversed(self.gives):                 # the last time it happened
                if all(ev[i] == g(r)[0] for r, i in want.items() if g(r) and i != out):
                    return ev[out]
            return None
        if pred in ("Q_REL_A1", "Q_REL_A2"):
            d = DIRS[g("D")[0]]
            for p, dd, q in self.rel:
                for s, e, o in ((p, dd, q), (q, INV[dd], p)):        # s is e of o
                    if e == d and pred == "Q_REL_A1" and o == g("A2")[0]:
                        return s
                    if e == d and pred == "Q_REL_A2" and s == x:
                        return o
            return None
        if pred == "Q_REL_YN":
            pos = self.positions(x)
            y, d = g("A2")[0], DIRS[g("D")[0]]
            if y not in pos:
                return "no"
            dx, dy = pos[x][0] - pos[y][0], pos[x][1] - pos[y][1]
            sx, sy = STEP[d]
            return "yes" if (dx * sx > 0 or dy * sy > 0) else "no"
        if pred == "Q_PATH":
            e, goal = self._edges(), g("A2")[0]
            seen, todo = {x}, deque([(x, [])])
            while todo:
                cur, path = todo.popleft()
                if cur == goal:
                    return ",".join(path)
                for d, nxt in e.get(cur, []):
                    if nxt not in seen:
                        seen.add(nxt)
                        todo.append((nxt, path + [d]))
            return None
        if pred == "Q_AFRAID":
            return self.afraid.get(self.kind.get(x))
        if pred == "Q_COLOR":
            if x in self.prop:
                return self.prop[x][-1][1]
            k = self.kind.get(x)                # induction: what the others of its kind are like
            same = [(self.kind_at[o], self.prop[o][-1][1]) for o, kk in self.kind.items()
                    if kk == k and o != x and o in self.prop]
            return max(same)[1] if same else None      # like the one of its kind that was introduced last
        if pred == "Q_FITS":
            return "yes" if (g("A2")[0], x) in self._closure() else "no"
        if pred == "Q_BIGGER":
            return "yes" if (x, g("A2")[0]) in self._closure() else "no"
        if pred == "Q_WHY":
            return self.prop[x][-1][1] if x in self.prop else None
        if pred == "Q_WILL":
            if x not in self.prop:
                return None
            feel = self.prop[x][-1][1]
            for o, vals in self.prop.items():   # someone who felt the same and then went somewhere
                for n, v in vals:
                    later = [m for m in self.moves.get(o, []) if m[1] > n]
                    if o != x and v == feel and later:
                        return later[0][2]
            return KB_GOES.get(feel)
        raise ValueError(pred)

    def positions(self, start):
        """coordinates of everything connected to start by the relations (start at 0, 0)"""
        e, pos, todo = self._edges(), {start: (0, 0)}, deque([start])
        while todo:
            cur = todo.popleft()
            for d, nxt in e.get(cur, []):
                if nxt not in pos:
                    pos[nxt] = (pos[cur][0] + STEP[d][0], pos[cur][1] + STEP[d][1])
                    todo.append(nxt)
        return pos


def well_formed(pred, args):
    """the type check of a frame: every role it needs is there, and nothing it cannot have"""
    need, may = NEEDS[pred], MAY.get(pred, ())
    return all(args.get(r) for r in need) and all(r in need or r in may for r in args if args[r])
