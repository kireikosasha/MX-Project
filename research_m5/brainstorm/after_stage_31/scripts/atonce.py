"""'Taken at once' on K, L, M2: who takes it - the memory word for word, or the network that sees no words?
For each task that the chain took at once: the stories as they are (first guess of rows.py only), 20 stories, ask 20.
  memory  - seed = what is remembered word for word only
  network - seed = what guess.pt proposes only
  both    - as in the blind run
Also: the share of the kinds of statements of the task that the memory holds word for word.
And the same with fewer stories (argv[3]) to see how 'at once' depends on the number of stories.
"""
import glob, json, os, sys, time, signal
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import rows, records, memory, guess

R = "/home/user/MX-Project/research_m5"
LOGIC = "/tmp/claude-0/-home-user-MX-Project/ace7a071-a9eb-520e-b8a4-5dce652d6be1/scratchpad/logic"
ASK = int(sys.argv[4]) if len(sys.argv) > 4 else 20
MEM = json.load(open(os.path.join(R, "memory.json")))


class Late(Exception):
    pass


def alarm(*_):
    raise Late()


signal.signal(signal.SIGALRM, alarm)


def tried(eps, stories, seed, tests, plain):
    if not seed:
        return "none"
    signal.alarm(40)
    try:
        m, used, notes = records.study(eps, stories, (), records.WORK, ASK, seed=seed, fast="only")
    except Late:
        return "late"
    finally:
        signal.alarm(0)
    if m is None:
        return "no"
    got, n = m.run(plain.rows(tests))
    return f"{100 * got / n:.0f}%"


def main():
    name, tasks, stories = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]) if len(sys.argv) > 3 else 20
    folder = os.path.join(LOGIC, "new", name)
    for k in tasks:
        raw = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*train.txt"))[0], stories + 200)
        tests = rows.stories(glob.glob(os.path.join(folder, f"qa{k}_*test.txt"))[0])
        plain = next(iter(rows.guesses(raw[:stories])))
        eps = plain.rows(raw)
        forms = {(op, quest, kinds) for ep in eps[:stories] for op, quest, _, _, kinds, _ in ep if not quest}
        seed = memory.recall(MEM, forms)
        t0 = time.time()
        net = guess.proposed(os.path.join(R, "guess.pt"), eps[:stories])
        out = [tried(eps, stories, [seed] if seed else [], tests, plain), tried(eps, stories, net, tests, plain), tried(eps, stories, ([seed] if seed else []) + net, tests, plain)]
        print(f"{name}{k:<3d} stories {stories:2d} | kinds of statements {len(forms)}, in memory word for word {len(seed)} | memory only: {out[0]:>5} | network only: {out[1]:>5} | both: {out[2]:>5} | {time.time() - t0:.0f} s", flush=True)


main()
