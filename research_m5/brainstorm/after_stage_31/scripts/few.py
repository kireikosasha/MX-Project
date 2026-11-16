"""Few stories: what holds the learner back - perception (which words are things), unseen kinds of questions, or the rest?
The search of recall.py with no words, from n stories with answers, no asking. Three perceptions:
  own    - the template reader built from the n stories only (as published)
  text   - the template reader built from the n stories and 200 more with the answers hidden (no answer more is used)
  oracle - the template reader as chosen from 20 stories with answers
For each: test accuracy; the share of the test questions whose kind was met in the n stories; accuracy on those.
"""
import glob, os, sys, time, signal
from collections import Counter
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import recall, rows

LOGIC = "/tmp/claude-0/-home-user-MX-Project/ace7a071-a9eb-520e-b8a4-5dce652d6be1/scratchpad/logic"


class Late(Exception):
    pass


def alarm(*_):
    raise Late()


signal.signal(signal.SIGALRM, alarm)


def pick(readers, labeled):
    """as recall.main chooses among the guesses about the words"""
    tried, plain = [], None
    for deep in (False, True):
        for reader in readers:
            for split in (False, True):
                m = recall.Learner(reader.rows(labeled), split, deep=deep)
                right, total = m.explained()
                plain = plain or len(m.readings)
                tried.append((right - 0.5 * len(m.readings) - m.cost() - 3 * bool(reader.guessed), right == total, id(reader), reader, split))
                if right == total or not m.task.split:
                    break
            if tried[-1][1] and len(m.readings) <= plain:
                break
        best = max(tried, key=lambda t: t[:3])
        if best[1]:
            break
    return best[3], best[4]


def one(readers, select, raw, n, tests):
    signal.alarm(90)
    try:
        reader, split = pick(readers, select)
        m, _, _ = recall.study(reader, raw[:n], n, 0, split)
        eps = reader.rows(tests)
        r, t, _, _ = m.test(eps)
        known = sum(y for x, y in m.tally.values())
        kinds = len({op for ep in reader.rows(raw[:n]) for op, quest, *_ in ep if not quest})
        return r, t, known, sum(x for x, y in m.tally.values()), kinds
    except Late:
        return None
    finally:
        signal.alarm(0)


def main():
    name, ns = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    folder = os.path.join(LOGIC, "new", name)
    for n in ns:
        for k in range(1, 13):
            raw = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], 240)
            tests = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*test.txt"))[0])[:150]
            hidden = [[(words, None if gold is None else []) for words, gold in story] for story in raw[40:240]]
            out = []
            for label, readers, select in (("own", list(rows.guesses(raw[:n])), raw[:n]),
                                           ("text", list(rows.guesses(raw[:n] + hidden)), raw[:n]),
                                           ("oracle", list(rows.guesses(raw[:20])), raw[:20])):
                got = one(readers, select, raw, n, tests)
                out.append("late" if got is None else f"{label} {got[0]}/{got[1]} known {got[2]} right {got[3]} kinds {got[4]}")
            print(f"{name} n={n} task {k}: " + " | ".join(out), flush=True)


main()
