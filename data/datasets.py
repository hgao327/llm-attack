import os
import json
import pandas as pd
import random
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

# =======================================
# 固定本地路径映射表
# =======================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_PATHS = {
    "sst2": os.path.join(BASE_DIR, "sst2"),  # 如果有的话
    "ag_news": os.path.join(BASE_DIR, "ag_news"),  # 如果有的话
    "c4": os.path.join(BASE_DIR, "processed_c4.json"),
    "eli5": os.path.join(BASE_DIR, "eli5.jsonl")  # 如果有的话
}

# 输出保存目录
PROCESSED_SAVE_DIR = "./datasets/processed_json"
os.makedirs(PROCESSED_SAVE_DIR, exist_ok=True)


# =======================================
# 各数据集特定解析函数
# =======================================
def load_sst2(path):
    dataset = load_from_disk(path)
    label_map = {0: "Negative", 1: "Positive"}
    formatted = []

    for example in dataset["train"]:
        text = example["sentence"]
        label = label_map[int(example["label"])]
        input_str = (
            # "Please classify the sentiment of the following text as either "
            # "'Positive' or 'Negative'. "
            f"Text: {text}. Sentiment:"
        )
        formatted.append({"input": input_str, "output": label, "text": text})
    return formatted


def load_agnews(path):
    train_csv = os.path.join(path, "train.csv")
    df = pd.read_csv(train_csv, header=None, names=["label", "title", "description"])
    df = df.dropna()
    label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    formatted = []

    for _, row in df.iterrows():
        text = f"{row['title']} {row['description']}"
        label = label_map[int(row["label"])]
        input_str = (
            # "Please classify the following news article into one of these "
            # "categories: World, Sports, Business, Sci/Tech. "
            f"Text: {text}. Category:"
        )
        formatted.append({"input": input_str, "output": label, "text": text})
    return formatted


def load_c4(path):
    dataset = load_dataset("json", data_files=path)
    data = dataset["train"]
    formatted = []

    for ex in data:
        prompt = ex.get("prompt", "") or ex.get("text", "")
        natural_text = ex.get("natural_text", "") or ex.get("target", "")
        input_str = f"Text: {prompt}\nAnswer:"
        output_str = natural_text
        formatted.append({"input": input_str, "output": output_str, "text": prompt})
    return formatted


def load_eli5(path):
    dataset = load_dataset("json", data_files=path)
    data = dataset["train"]
    formatted = []

    for ex in data:
        q = ex.get("title", "") or ex.get("question_title", "")
        a = ""
        if "C1" in ex and isinstance(ex["C1"], dict):
            comment_list = ex["C1"].get("comment_text", [])
            if isinstance(comment_list, list) and comment_list:
                a = comment_list[0]
        elif "answers" in ex:
            answers = ex.get("answers")
            if isinstance(answers, list) and answers:
                a = answers[0]
        input_str = f"Question: {q}\nAnswer:"
        formatted.append({"input": input_str, "output": a, "text": q})
    return formatted


# =======================================
# 保存为 JSON 的工具函数
# =======================================
def save_to_json(data, name):
    """
    将格式化好的数据列表保存为 JSON 文件。
    """
    save_path = os.path.join(PROCESSED_SAVE_DIR, f"{name}_processed.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[datasets.py] Saved processed dataset → {save_path}")


# =======================================
# 统一接口
# =======================================
def load_texts(
    dnames,
    samples_per_ds=1000,
    seed=42,
):
    """
    统一加载接口：
    - 支持 sst2, ag_news, c4, eli5
    - 自动根据 LOCAL_PATHS 定位文件路径
    - 返回 List[str]（仅 input 字段，供 probe 使用）
    - 同时保存完整样本结构至 processed_json 文件夹
    """
    random.seed(seed)
    all_data = []

    for name in dnames:
        key = name.lower().replace("-", "_")
        if key not in LOCAL_PATHS:
            raise ValueError(f"[datasets.py] Unknown dataset name: {name}")
        path = LOCAL_PATHS[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"[datasets.py] Path not found: {path}")

        print(f"[datasets.py] Loading local dataset: {key} from {path}")
        if key == "sst2":
            data = load_sst2(path)
        elif key in ["ag_news", "agnews"]:
            data = load_agnews(path)
        elif key == "c4":
            data = load_c4(path)
        elif key == "eli5":
            data = load_eli5(path)
        else:
            raise ValueError(f"Unsupported dataset type: {key}")

        # 随机采样并保存
        if samples_per_ds > 0 and len(data) > samples_per_ds:
            data = random.sample(data, samples_per_ds)

        # 保存至本地 JSON
        save_to_json(data, key)

        all_data.extend(data)

    print(f"[datasets.py] Total samples loaded: {len(all_data)}")
    return [x["input"] for x in all_data]

def load_full_data(
    dnames,
    samples_per_ds=1000,
    seed=42,
):
    """
    加载完整样本结构（含 input + output + text）
    与 load_texts() 不同：返回 List[Dict]，用于带监督任务。
    """
    random.seed(seed)
    all_data = []

    for name in dnames:
        key = name.lower().replace("-", "_")
        if key not in LOCAL_PATHS:
            raise ValueError(f"[datasets.py] Unknown dataset name: {name}")
        path = LOCAL_PATHS[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"[datasets.py] Path not found: {path}")

        print(f"[datasets.py] Loading local dataset: {key} from {path}")
        if key == "sst2":
            data = load_sst2(path)
        elif key in ["ag_news", "agnews"]:
            data = load_agnews(path)
        elif key == "c4":
            data = load_c4(path)
        elif key == "eli5":
            data = load_eli5(path)
        else:
            raise ValueError(f"Unsupported dataset type: {key}")

        # 随机采样
        if samples_per_ds > 0 and len(data) > samples_per_ds:
            data = random.sample(data, samples_per_ds)

        # 保存副本
        save_to_json(data, key)
        all_data.extend(data)

    print(f"[datasets.py] Total full samples loaded: {len(all_data)}")
    return all_data
