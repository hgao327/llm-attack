from datasets import load_dataset

# ========== SST-2 ==========
def dump_sst2(n=1000):
    ds = load_dataset("glue", "sst2", split="train")
    with open("sst2_clean.tsv", "w", encoding="utf-8") as f:
        for i in range(n):
            text = ds[i]["sentence"].replace("\t", " ").strip()
            label = ds[i]["label"]   # 0 = negative, 1 = positive
            f.write(f"{label}\t{text}\n")

# ========== AGNews ==========
def dump_agnews(n=1000):
    ds = load_dataset("ag_news", split="train")
    with open("agnews_clean.tsv", "w", encoding="utf-8") as f:
        for i in range(n):
            text = ds[i]["text"].replace("\t", " ").strip()
            label = ds[i]["label"]   # 0/1/2/3
            f.write(f"{label}\t{text}\n")

dump_sst2()
dump_agnews()
print("clean data saved")
