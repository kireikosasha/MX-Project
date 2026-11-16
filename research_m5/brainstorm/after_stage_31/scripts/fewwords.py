"""Few stories with the perception fixed by unlabeled text: does the reader of words now beat the search with no words?
Perception: the template reader built from the n stories and 200 more with the answers hidden.
  search - recall.study on the n stories (no words)
  words0 - the choices read from the words alone (no step), default and turn fitted on the n stories
  words  - then 100 steps of the gradient on the n stories and the exact finish (as soft.py does)
Test: 150 test stories.
"""
import glob, os, sys, signal
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import torch
import recall, rows, soft

torch.set_num_threads(1)
LOGIC = "/tmp/claude-0/-home-user-MX-Project/ace7a071-a9eb-520e-b8a4-5dce652d6be1/scratchpad/logic"


class Late(Exception):
    pass


def alarm(*_):
    raise Late()


signal.signal(signal.SIGALRM, alarm)


def tested(reader, task, readings, tests):
    m = recall.Learner.__new__(recall.Learner)
    m.task, m.readings = task, readings
    m.count = {q: len(task.golds(q)) for q in task.asked.values()}
    m.usual = {q: (recall.Counter(task.golds(q)).most_common(1) or [("", 0)])[0][0] for q in task.asked.values()}
    m.own = {q: task.own(q) for q in task.asked.values()}
    r, t, _, _ = m.test(reader.rows(tests))
    return r, t


def main():
    name, ns, wpath = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    saved = torch.load(wpath)
    words = soft.Reader(saved["vocab"], whole=saved.get("whole", False))
    words.load_state_dict(saved["weights"])
    folder = os.path.join(LOGIC, "new", name)
    for n in ns:
        for k in range(1, 13):
            raw = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], 240)
            tests = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*test.txt"))[0])[:150]
            hidden = [[(w, None if gold is None else []) for w, gold in story] for story in raw[40:240]]
            signal.alarm(150)
            try:
                tried = []
                for reader in rows.guesses(raw[:n] + hidden):
                    task = recall.Task(reader.rows(raw[:n]), False)
                    if max(sum(c != "#" for c in f[2]) for f in list(task.forms) + list(task.asked)) > soft.SLOTS:
                        continue
                    task = soft.widened(reader, task, tests)
                    layer = soft.Layer(soft.Stories(task))
                    with torch.no_grad():
                        layer.given = words.choices(task, layer)
                        readings = layer.hard(task)
                    right, total = sum(r.right for r in readings.values()), sum(r.total for r in readings.values())
                    tried.append((right - 0.5 * len(readings) - 3 * bool(reader.guessed), id(reader), reader, task, readings, right, total, layer.given))
                    if right == total:
                        break
                if not tried:
                    print(f"{name} n={n} task {k}: more than three things", flush=True)
                    continue
                _, _, reader, task, readings, right, total, given = max(tried)
                w0 = tested(reader, task, readings, tests)
                if right < total:
                    readings, right, total = soft.learn(task, 100, 1, start=given)
                if right < total:
                    readings = soft.polished(task, readings)
                w1 = tested(reader, task, readings, tests)
                m, _, _ = recall.study(reader, raw[:n], n, 0, False)
                s = m.test(reader.rows(tests))[:2]
                print(f"{name} n={n} task {k}: search {s[0]}/{s[1]} | words0 {w0[0]}/{w0[1]} | words {w1[0]}/{w1[1]}", flush=True)
            except Late:
                print(f"{name} n={n} task {k}: late", flush=True)
            finally:
                signal.alarm(0)


main()
