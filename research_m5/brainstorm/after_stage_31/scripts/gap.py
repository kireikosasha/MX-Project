import re, os
from collections import Counter
R = "/home/user/MX-Project/research_m5/results"
SAY = {"set","as","lift","clear","swap","link","two"}
ASK = {"value","end","second","holders","inside","after","before","is","isin","same","any","known","leads","count","countin","countafter","countbefore"}
def recall_file(path):
    cur=None; tasks={}
    for line in open(path):
        m=re.match(r"task +(\d+):",line)
        if m: cur=int(m.group(1)); tasks[cur]={"roles":set(),"asks":set(),"test":None,"sorts":set()}; continue
        m=re.match(r"test +(\d+): (\d+)/(\d+)",line)
        if m and int(m.group(1)) in tasks: tasks[int(m.group(1))]["test"]=int(m.group(2))/max(1,int(m.group(3))); continue
        m=re.match(r"  \? (.*?): (\d+)/(\d+) (value|link|amount): (.*?) \| (.*)$",line.rstrip("\n"))
        if m and cur is not None:
            ask=re.sub(r"\(turned over\)|\(else [^)]*\)","",m.group(5)).strip()
            tasks[cur]["asks"]|={t for t in ask.split() if t.isalpha()}
            tasks[cur]["sorts"].add(m.group(4))
            for part in m.group(6).split("; "):
                if " = " in part:
                    tasks[cur]["roles"]|=set(re.findall(r"[a-z]+",part.rsplit(" = ",1)[1]))
    return tasks
def soft_file(path):
    out={}
    for line in open(path):
        m=re.match(r"task +(\d+): its sentences have more",line)
        if m: out[int(m.group(1))]="skipped"; continue
        m=re.match(r"test +(\d+): (\d+)/(\d+)",line)
        if m: out[int(m.group(1))]=int(m.group(2))/max(1,int(m.group(3)))
    return out
cat=Counter(); lists={}
for name in ["en","A","B","C","D","E","F","G","H","I","J","K","L"]:
    rec=recall_file(os.path.join(R,"recall",("L_after_the_blind_run" if name=="L" else name)+".txt"))
    sof=soft_file(os.path.join(R,"soft",name+".txt"))
    for k,t in rec.items():
        s=sof.get(k)
        if s is None or t["test"] is None: continue
        inside = t["roles"]<=SAY and t["asks"]<=ASK and "amount" not in t["sorts"]
        if s=="skipped": c="skipped by the layer (more than 3 things)" + (", search 95+" if t["test"]>=.95 else "")
        elif s>=.95: c="layer 95+" + (" (search below 95)" if t["test"]<.95 else "")
        elif t["test"]>=.95: c="layer below 95, search 95+: " + ("reading INSIDE the layer's parts (the gradient did not find it)" if inside else "reading outside the layer's parts")
        else: c="both below 95"
        cat[c]+=1; lists.setdefault(c,[]).append(f"{name}{k}" + ("" if s=="skipped" else f"({100*s:.0f})"))
for c,n in cat.most_common(): print(n,c,"|"," ".join(lists[c])[:330])
