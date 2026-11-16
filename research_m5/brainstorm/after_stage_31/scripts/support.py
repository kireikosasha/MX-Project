import re, glob, os, sys
from collections import Counter
R = "/home/user/MX-Project/research_m5/results/recall"
files = ["en","A","B","C","D","E","F","G","H","I","J","K","L_after_the_blind_run","blind_M2"]
SAY = {"set","clear","as","link","unlink","put","add","take","pass"}
ASK = {"end","value","holders","count","is","same","leads","after","before","countafter","last","amount","morethan","most","more"}
WR = ("asof","until","was","began","prior","upto","ere")
tot=Counter(); roleC=Counter(); askC=Counter()
rows=[]
for name in files:
    cur=None; tasks={}
    for line in open(os.path.join(R,name+".txt")):
        m=re.match(r"task +(\d+):",line)
        if m: cur=int(m.group(1)); tasks[cur]={"q":[], "test":None}; continue
        m=re.match(r"test +(\d+): (\d+)/(\d+)",line)
        if m: tasks[int(m.group(1))]["test"]=int(m.group(2))/max(1,int(m.group(3))); continue
        m=re.match(r"  \? (.*?): (\d+)/(\d+) (value|link|amount): (.*?) \| (.*)$",line.rstrip("\n"))
        if m and cur is not None:
            sort=m.group(4); ask=m.group(5); roles=m.group(6)
            ask=re.sub(r"\(turned over\)|\(else [^)]*\)","",ask).strip()
            atoks=ask.split()
            anames=[t for t in atoks if t.isalpha()]
            rnames=set()
            for part in roles.split("; "):
                if " = " in part:
                    r=part.rsplit(" = ",1)[1]
                    rnames |= set(re.findall(r"[a-z]+",r))
            tasks[cur]["q"].append((int(m.group(2)),int(m.group(3)),sort,anames,rnames))
    for k,t in tasks.items():
        if t["test"] is None: continue
        used_roles=set(); used_asks=set()
        for r,tt,sort,an,rn in t["q"]:
            used_roles|=rn; used_asks|=set(an)
        ok = used_roles<=SAY and used_asks<=ASK
        rows.append((name,k,t["test"],ok,used_roles-SAY,used_asks-ASK))
        if t["test"]>=0.95:
            for x in used_roles: roleC[x]+=1
            for x in used_asks: askC[x]+=1
n95=[r for r in rows if r[2]>=0.95]
print("tasks with test:",len(rows),"at 95+:",len(n95))
print("at 95+ whose whole reading is inside the support of drawn() (by part/ask names):",sum(r[3] for r in n95))
print("all tasks inside support:",sum(r[3] for r in rows))
print("roles used in 95+ tasks:",roleC.most_common())
print("asks used in 95+ tasks:",askC.most_common())
out=Counter()
for r in n95:
    for x in r[4]|r[5]: out[x]+=1
print("what is outside, in how many 95+ tasks:",out.most_common())
