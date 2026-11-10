#!/usr/bin/env python3
"""
Decision thresholds of the chat model chosen on the validation set (as the Millennium library selects its
decision threshold: on validation data, not recalibrated on the training set).

  wiki   P([W]) from which the question goes to the Wikipedia reader: the threshold with the best F1
  sure   P([W]) from which a failed search ends in "I don't know": the lowest threshold with precision >= 0.97
  conf   mean log-probability of the tokens of a reply below which the model says it is not sure. It is a
         setting, not measured here: the replies the model writes itself are more likely than the replies of
         the validation set, so their percentiles say little. -2.5 comes from 220 replies of the published
         model (chat_ref.py): below it the replies were nonsense, about a fifth of the replies to real-life
         questions. The percentiles of the validation replies are printed for reference.

    python chat_thresholds.py --model out_ft/chat.pt --data data --out out_ft/thresholds.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_train as CT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--conf", type=float, default=-2.5)
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    sp = ck["special"]
    m = CT.ChatLM(ck["V"], ck["classes"], **ck["arch"])
    m.load_state_dict(ck["state"])
    m.to(a.device).eval()
    m.members = [t.to(a.device) for t in m.members]
    data = CT.Data(a.data, sp, a.device)
    va = np.where(data.split == 1)[0]
    cls = m.cls_of.cpu().numpy()
    va = va[~((data.M[va, :-1] == 1) & (cls[data.X[va, 1:]] < 0)).any(1)][:a.limit]
    CT.LEVELS = ck.get("levels", 127)
    emb = CT.quantize(m.emb.detach(), float(ck["scale"]))
    W, B = sp["[W]"], sp["[B]"]
    p_w, lab, conf = [], [], []
    with torch.no_grad():
        for i in range(0, len(va), 256):
            x, mk, r = data.batch(va[i:i + 256])
            lp, pw, dec = CT.log_probs(m, x, mk, r, emb, W)
            rows, cols = dec.nonzero(as_tuple=True)
            first = x[rows, cols] == B                      # the position that decides: reply or [W]
            y = x[:, 1:][dec]
            p_w += torch.exp(pw[first]).tolist()
            lab += (y[first] == W).tolist()
            keep = y != W
            tot = torch.zeros(x.shape[0], device=x.device).index_add_(0, rows[keep], lp[keep])
            cnt = torch.zeros(x.shape[0], device=x.device).index_add_(0, rows[keep], torch.ones_like(lp[keep]))
            conf += (tot[cnt > 0] / cnt[cnt > 0]).tolist()
    p_w, lab = np.array(p_w), np.array(lab, bool)
    best, wiki, sure = -1.0, 0.1, 0.4
    for t in np.arange(0.02, 0.96, 0.02):
        tp, fp, fn = (lab & (p_w >= t)).sum(), (~lab & (p_w >= t)).sum(), (lab & (p_w < t)).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best:
            best, wiki = f1, float(t)
    for t in np.arange(wiki, 0.96, 0.02):
        tp, fp = (lab & (p_w >= t)).sum(), (~lab & (p_w >= t)).sum()
        sure = float(t)
        if tp / max(1, tp + fp) >= 0.97:
            break
    out = {"wiki": round(wiki, 2), "sure": round(max(sure, wiki), 2), "conf": a.conf,
           "val_conf_p1_p50": [round(float(np.percentile(conf, q)), 3) for q in (1, 50)] if conf else [],
           "f1": round(float(best), 4), "n": int(len(lab)), "wiki_share": round(float(lab.mean()), 4)}
    json.dump(out, open(a.out, "w"))
    print("thresholds chosen on the validation set:", out)


if __name__ == "__main__":
    main()
