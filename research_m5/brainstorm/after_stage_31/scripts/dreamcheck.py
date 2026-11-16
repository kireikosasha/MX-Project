"""the ground's own search on the ground's own dreams (the ceiling for a network taught on them), and what a
predictor that only knows the usual answers of a kind of question gets (the floor mind.py's numbers are to be read against)"""
import sys, time, random
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/MX-Project/research_m5")
from collections import Counter
import recall

rng = random.Random(5)
gen = recall.dreams(rng, 40)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
full = at95 = 0; tr = tt = 0; t0 = time.time()
maj_r = maj_t = 0; tok_r = tok_t = 0; kinds_q = kinds_s = 0; shown = 0
for i in range(n):
    eps, _, _ = next(gen)
    if len(eps) < 30:
        continue
    shown += 1
    train, test = eps[:20], eps[20:]
    m = recall.Learner(train)
    r, t = m.explained()
    full += r == t
    a, b, _, _ = m.test(test)
    tr += a; tt += b; at95 += a >= 0.95 * b
    kinds_q += len(m.task.asked); kinds_s += len(m.task.forms)
    usual, nxt = {}, {}
    for ep in train:
        for op, quest, things, numbers, ks, gold in ep:
            if quest:
                usual.setdefault(op, Counter())[gold] += 1
                toks = sorted(gold.split(",")) + ["<end>"]
                for j, w in enumerate(toks):
                    nxt.setdefault((op, j), Counter())["<end>" if w == "<end>" else ("thing" if recall.thing(w) else w)] += 1
    for ep in test:
        for op, quest, things, numbers, ks, gold in ep:
            if quest:
                maj_t += 1
                maj_r += bool(usual.get(op)) and usual[op].most_common(1)[0][0] == gold
                toks = sorted(gold.split(",")) + ["<end>"]
                for j, w in enumerate(toks):
                    tok_t += 1
                    c = nxt.get((op, j))
                    guess = c.most_common(1)[0][0] if c else "<end>"
                    # a thing guessed blindly: one chance in as many things as the task has of that kind (about 5)
                    tok_r += (guess == w) if guess != "thing" else (0.2 if recall.thing(w) else 0)
print(f"{shown} dreamt tasks, 20 stories to learn from, 20 to check on; on the average {kinds_s / shown:.1f} kinds of statements, {kinds_q / shown:.1f} kinds of questions")
print(f"the search of recall.py on them: explains all its own questions in {full} tasks; on the check {100 * tr / tt:.1f}% of the answers, {at95} tasks at 95% and more; {(time.time() - t0) / shown:.2f} s a task")
print(f"the usual answer of the kind of question (learnt from the same 20 stories): whole answers {100 * maj_r / maj_t:.1f}%")
print(f"the usual next word of an answer by its place (yes/no, numbers, the end; a thing guessed blindly, 1 in 5): words {100 * tok_r / tok_t:.1f}%")
