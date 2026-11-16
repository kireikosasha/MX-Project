import glob, json, os, sys
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import rows, memory
R = "/home/user/MX-Project/research_m5"
LOGIC = "/tmp/claude-0/-home-user-MX-Project/ace7a071-a9eb-520e-b8a4-5dce652d6be1/scratchpad/logic"
MEM = json.load(open(os.path.join(R, "memory.json")))
taken = {"K": {2, 3, 9, 12}, "L": {2, 3, 7, 8, 9, 10}, "M2": {2, 3, 4, 8, 10, 12}}
tot = {True: [0, 0, 0, 0], False: [0, 0, 0, 0]}
for name in ("K", "L", "M2"):
    line = []
    for k in range(1, 13):
        raw = rows.stories(glob.glob(os.path.join(LOGIC, "new", name, f"qa{k}_*train.txt"))[0], 20)
        plain = next(iter(rows.guesses(raw)))
        eps = plain.rows(raw)
        forms = {(op, quest, kinds) for ep in eps for op, quest, _, _, kinds, _ in ep if not quest}
        seed = memory.recall(MEM, forms)
        t = tot[k in taken[name]]
        t[0] += len(seed); t[1] += len(forms); t[2] += len(seed) == len(forms); t[3] += 1
        line.append(f"{k}:{len(seed)}/{len(forms)}{'*' if k in taken[name] else ''}")
    print(name, " ".join(line))
for key, t in tot.items():
    print("taken at once" if key else "not taken", f"kinds of statements in memory word for word {t[0]}/{t[1]} = {100*t[0]/t[1]:.0f}%; tasks with every kind in memory {t[2]}/{t[3]}")
