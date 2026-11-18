import sys, random, ast, csv
sys.path.insert(0, "/home/user/MX-Project/research_m5")
import walk
D = sys.argv[1]
kinds, asks, answers = {}, {}, {}
taught = walk.kin(f"{D}/gen23_train.csv", kinds, asks, answers, True)
test = walk.kin(f"{D}/gen23_test.csv", kinds, asks, answers, False)
names = {i: k for k, i in kinds.items()}
print("test stories", len(test), "by links:", sorted({x[3] for x in test}))
for n in (30, 60, 100, 200, 400, 9074):
    rng = random.Random(0)
    sub = rng.sample(taught, min(n, len(taught)))
    t = walk.Table([(s, q, g) for s, q, g, _ in sub if s], names, answers)
    by = {}
    for s, q, g, k in test:
        ok = s is not None and t.hard(s, q) == g
        by.setdefault(k, []).append(ok)
    print(f"table from {n:5d} stories: all {100*sum(map(sum, by.values()))/len(test):.1f}")
