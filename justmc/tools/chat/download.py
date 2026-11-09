#!/usr/bin/env python3
"""
Downloads every dataset the reader and the chat model are trained on (Hugging Face), into one folder:

  QA (reader, [W] questions of the chat model)
    squad_train / squad_val.parquet     SQuAD 2.0 (rajpurkar/squad_v2)               en
    sber_train / sber_val.parquet       SberQuAD (kuznetsoffandrey/sberquad)         ru
    uahplt_train / uahplt_test.parquet  UA-SQuAD (HPLT/ua-squad)                     uk
    fido_train / fido_val.json          UA-SQuAD (FIdo-AI/ua-squad)                  uk
    xquad_ru.parquet                    XQuAD ru (google/xquad), check only
  dialogues (chat model)
    dd_train.zip                        DailyDialog (roskoN/dailydialog)             en
    persona.parquet                     PersonaChat (AlekseyKorshuk/persona-chat)    en
    everyday.parquet                    everyday-conversations (HuggingFaceTB)       en
    rudialogues.jsonl                   russian_dialogues (Den4ikAI), the first --ru-mb MB
    aya.parquet                         Aya dataset (CohereLabs)                     uk part

    python download.py --out raw [--ru-mb 200]
Files that are already there are skipped.
"""
import argparse
import os
import sys
import urllib.request

API = "https://huggingface.co/api/datasets/"
HF = "https://huggingface.co/datasets/"
FILES = {
    "squad_train.parquet": API + "rajpurkar/squad_v2/parquet/squad_v2/train/0.parquet",
    "squad_val.parquet": API + "rajpurkar/squad_v2/parquet/squad_v2/validation/0.parquet",
    "sber_train.parquet": API + "kuznetsoffandrey/sberquad/parquet/sberquad/train/0.parquet",
    "sber_val.parquet": API + "kuznetsoffandrey/sberquad/parquet/sberquad/validation/0.parquet",
    "uahplt_train.parquet": API + "HPLT/ua-squad/parquet/default/train/0.parquet",
    "uahplt_test.parquet": API + "HPLT/ua-squad/parquet/default/test/0.parquet",
    "fido_train.json": HF + "FIdo-AI/ua-squad/resolve/main/train.json",
    "fido_val.json": HF + "FIdo-AI/ua-squad/resolve/main/val.json",
    "xquad_ru.parquet": API + "google/xquad/parquet/xquad.ru/validation/0.parquet",
    "dd_train.zip": HF + "roskoN/dailydialog/resolve/main/train.zip",
    "persona.parquet": HF + "AlekseyKorshuk/persona-chat/resolve/main/data/train-00000-of-00001.parquet",
    "everyday.parquet": HF + "HuggingFaceTB/everyday-conversations-llama3.1-2k/resolve/main/data/train_sft-00000-of-00001.parquet",
    "rudialogues.jsonl": HF + "Den4ikAI/russian_dialogues/resolve/main/dataset.jsonl",
    "aya.parquet": HF + "CohereLabs/aya_dataset/resolve/main/data/train-00000-of-00001.parquet",
}


def fetch(url, path, limit=None):
    req = urllib.request.Request(url)
    if limit:
        req.add_header("Range", f"bytes=0-{limit - 1}")
    tmp = path + ".part"
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
        n = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            out.write(b)
            n += len(b)
            print(f"\r  {os.path.basename(path)}: {n / 1e6:.0f} MB", end="", flush=True)
    os.replace(tmp, path)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ru-mb", type=int, default=200, help="MB of russian_dialogues to take (the whole file is 1.4 GB)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(a.out, name)
        if os.path.exists(path):
            print(f"  {name}: already there")
            continue
        try:
            fetch(url, path, a.ru_mb * 1000000 if name == "rudialogues.jsonl" else None)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"{name}: {e}")


if __name__ == "__main__":
    main()
