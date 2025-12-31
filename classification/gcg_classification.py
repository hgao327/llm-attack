import os
import sys

# 必须在 import torch 之前设置环境变量，禁用 MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

# Force CPU to avoid MPS bus errors
DEVICE = torch.device("cpu")

# Aggressively disable MPS
if hasattr(torch, 'has_mps'):
    torch.has_mps = False
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False


import platform
print(platform.platform())
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")



# ============ LLM-Attacks GCG 实现 ============
def token_gradients(model, input_ids, attention_mask, target_label):
    """
    计算每个 token 位置的梯度
    基于 LLM-Attacks 的实现
    """
    # 获取 embedding layer
    embed_layer = model.get_input_embeddings()

    # 获取 one-hot 编码
    one_hot = torch.zeros(
        input_ids.shape[0],
        input_ids.shape[1],
        embed_layer.weight.shape[0],
        dtype=torch.float32,
        device=DEVICE
    )
    one_hot.scatter_(2, input_ids.unsqueeze(2), 1.0)
    one_hot.requires_grad_(True)

    # 通过 one-hot 获取 embeddings
    embeds = torch.matmul(one_hot, embed_layer.weight)

    # 前向传播
    logits = model(inputs_embeds=embeds, attention_mask=attention_mask).logits

    # 计算损失（我们想要最大化损失以翻转预测）
    loss = F.cross_entropy(logits, target_label)

    # 反向传播
    loss.backward()

    # 返回梯度
    return one_hot.grad.clone()


def sample_control(control_toks, grad, num_candidates, topk=256):
    """
    基于梯度采样候选 token
    实现 LLM-Attacks 的 GCG 采样策略

    目标：最大化 loss（即增加模型对真实标签的损失）
    梯度方向：grad 指向 loss 增加的方向
    """
    if len(grad.shape) == 3:
        # [batch_size, seq_len, vocab_size] -> [seq_len, vocab_size]
        grad = grad.mean(dim=0)

    # 为每个位置获取 top-k 最优替换
    # 选择梯度最大的 tokens（沿梯度正方向 = 最大化 loss）
    top_indices = grad.topk(topk, dim=-1).indices  # [seq_len, topk]

    # 生成候选
    control_toks_repeated = control_toks.repeat(num_candidates, 1)

    for i in range(num_candidates):
        # 随机选择一个位置进行修改
        pos = np.random.randint(0, control_toks.shape[1])
        # 从该位置的 top-k 中随机选择一个 token
        new_token_idx = np.random.randint(0, topk)
        control_toks_repeated[i, pos] = top_indices[pos, new_token_idx]

    return control_toks_repeated


def gcg_attack(
    model,
    tokenizer,
    text,
    label,
    num_steps=100,
    adv_string_init="! ! ! ! ! !",
    num_candidates=256,
    topk=256,
    position='suffix'
):
    """
    使用 GCG (Greedy Coordinate Gradient) 方法生成对抗样本
    基于 "Universal and Transferable Adversarial Attacks on Aligned Language Models"

    Args:
        model: 目标模型
        tokenizer: tokenizer
        text: 原始文本
        label: 真实标签
        num_steps: GCG 优化步数
        adv_string_init: 对抗字符串初始化
        num_candidates: 每步生成的候选数量
        topk: 每个位置考虑的 top-k tokens
        position: 对抗字符串位置 ('suffix' 或 'prefix')
    """
    model.eval()
    model = model.cpu()

    # 原始预测
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.cpu() for k, v in inputs.items()}
        orig_logits = model(**inputs).logits
        orig_pred = orig_logits.argmax(dim=-1).item()

    # 如果原始预测就是错的，直接返回
    if orig_pred != label:
        return text

    # 初始化对抗字符串
    adv_tokens = tokenizer(adv_string_init, add_special_tokens=False, return_tensors="pt")["input_ids"]
    adv_tokens = adv_tokens.cpu()

    # Tokenize 原始文本
    text_tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    text_tokens = text_tokens.cpu()

    best_adv_tokens = adv_tokens.clone()
    best_loss = float('-inf')

    for step in range(num_steps):
        # 构建完整输入
        if position == 'suffix':
            input_ids = torch.cat([
                torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).cpu(),
                text_tokens,
                adv_tokens,
                torch.tensor([[tokenizer.sep_token_id]], dtype=torch.long).cpu()
            ], dim=1)
            adv_slice = slice(1 + text_tokens.shape[1], 1 + text_tokens.shape[1] + adv_tokens.shape[1])
        else:
            input_ids = torch.cat([
                torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).cpu(),
                adv_tokens,
                text_tokens,
                torch.tensor([[tokenizer.sep_token_id]], dtype=torch.long).cpu()
            ], dim=1)
            adv_slice = slice(1, 1 + adv_tokens.shape[1])

        attention_mask = torch.ones_like(input_ids).cpu()
        target_label = torch.tensor([label], dtype=torch.long).cpu()

        # 计算梯度
        grad = token_gradients(model, input_ids, attention_mask, target_label)

        # 只取对抗 tokens 的梯度
        adv_grad = grad[:, adv_slice, :]

        # 采样候选
        adv_token_candidates = sample_control(
            adv_tokens,
            adv_grad,
            num_candidates,
            topk=topk
        )

        # 评估所有候选
        best_candidate = adv_tokens.clone()
        best_candidate_loss = best_loss

        for candidate in adv_token_candidates:
            # 构建完整输入
            if position == 'suffix':
                candidate_input_ids = torch.cat([
                    torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).cpu(),
                    text_tokens,
                    candidate.unsqueeze(0),
                    torch.tensor([[tokenizer.sep_token_id]], dtype=torch.long).cpu()
                ], dim=1)
            else:
                candidate_input_ids = torch.cat([
                    torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).cpu(),
                    candidate.unsqueeze(0),
                    text_tokens,
                    torch.tensor([[tokenizer.sep_token_id]], dtype=torch.long).cpu()
                ], dim=1)

            candidate_attention_mask = torch.ones_like(candidate_input_ids).cpu()

            # 评估
            with torch.no_grad():
                logits = model(
                    input_ids=candidate_input_ids,
                    attention_mask=candidate_attention_mask
                ).logits
                loss = F.cross_entropy(logits, target_label).item()
                pred = logits.argmax(dim=-1).item()

            # 检查是否成功翻转预测
            if pred != label:
                # 成功！返回结果
                adv_string = tokenizer.decode(candidate, skip_special_tokens=True)
                if position == 'suffix':
                    return text + " " + adv_string
                else:
                    return adv_string + " " + text

            # 更新最佳候选
            if loss > best_candidate_loss:
                best_candidate_loss = loss
                best_candidate = candidate.clone()

        # 更新对抗 tokens
        if best_candidate_loss > best_loss:
            best_loss = best_candidate_loss
            adv_tokens = best_candidate.unsqueeze(0)
            best_adv_tokens = best_candidate.clone()

    # 如果没有成功翻转，返回最佳尝试
    adv_string = tokenizer.decode(best_adv_tokens, skip_special_tokens=True)
    if position == 'suffix':
        return text + " " + adv_string
    else:
        return adv_string + " " + text


# ============ 数据集统一接口 ============
def get_dataset(name):
    if name == "sst2":
        ds = load_dataset("glue", "sst2", split="train")
        return [(x["sentence"], x["label"]) for x in ds]

    if name == "agnews":
        ds = load_dataset("ag_news", split="train")
        return [(x["text"], x["label"]) for x in ds]

    raise ValueError(name)


# ============ 主生成逻辑 ============
def generate_file(
    dataset_name,
    model_name,
    out_path,
    num_samples=1000,
    num_steps=50,
    adv_string_init="! ! ! ! ! !",
    num_candidates=64,
    topk=256,
):
    """
    使用 GCG 方法生成对抗样本数据集

    Args:
        dataset_name: 数据集名称
        model_name: 模型名称
        out_path: 输出路径
        num_samples: 生成样本数量
        num_steps: GCG 优化步数
        adv_string_init: 对抗字符串初始化
        num_candidates: 每步候选数量
        topk: top-k 采样
    """
    # 标签映射
    if dataset_name == "sst2":
        label_names = {0: "negative", 1: "positive"}
    elif dataset_name == "agnews":
        label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    else:
        label_names = {i: f"label_{i}" for i in range(10)}

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).cpu()

    print(f"Loading dataset: {dataset_name}")
    data = get_dataset(dataset_name)
    random.shuffle(data)
    # 只取需要的样本数量
    data = data[:num_samples]

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    f = open(out_path, "w", encoding="utf-8")

    success_count = 0
    cnt = 0

    for text, label in tqdm(data, desc=f"Generating {dataset_name} with GCG", total=num_samples):
        if cnt >= num_samples:
            break

        adv_text = gcg_attack(
            model,
            tokenizer,
            text,
            label,
            num_steps=num_steps,
            adv_string_init=adv_string_init,
            num_candidates=num_candidates,
            topk=topk,
            position='suffix'
        )

        # 检查是否成功
        with torch.no_grad():
            inputs = tokenizer(adv_text, return_tensors="pt", truncation=True)
            inputs = {k: v.cpu() for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred = logits.argmax(dim=-1).item()

        if pred != label:
            success_count += 1

        # 写入新格式
        label_name = label_names.get(label, f"label_{label}")
        f.write(f"Sample {cnt + 1}:\n")
        f.write(f"    Original: {text.replace(chr(10), ' ')}\n")
        f.write(f"    Adversarial: {adv_text.replace(chr(10), ' ')}\n")
        f.write(f"    Label: {label_name}\n")
        f.write("\n")

        cnt += 1

        # 每10个样本刷新一次，避免程序中断时丢失数据
        if cnt % 10 == 0:
            f.flush()

        if cnt % 50 == 0:
            success_rate = success_count / cnt * 100
            print(f"  Progress: {cnt}/{num_samples}, Success rate: {success_rate:.2f}%")

    f.close()

    final_success_rate = success_count / cnt * 100 if cnt > 0 else 0
    print(f"\n✓ Saved {cnt} samples to {out_path}")
    print(f"✓ Attack success rate: {final_success_rate:.2f}% ({success_count}/{cnt})")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GCG Attack (LLM-Attacks Method)")
    print("="*70)

    # Create output dir (in parent)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # SST-2 数据集
    print("\n[1/2] Generating SST-2 adversarial samples...")
    generate_file(
        dataset_name="sst2",
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        out_path=os.path.join(output_dir, "gcg_sst2.txt"),
        num_samples=1000,
        num_steps=50,                # GCG optimization steps
        adv_string_init="! ! ! ! ! !",  # Initial adversarial string (6 tokens)
        num_candidates=64,           # Candidates per step (reduce for speed)
        topk=256,                    # top-k 采样
    )

    # AG News 数据集
    print("\n[2/2] Generating AG News adversarial samples...")
    generate_file(
        dataset_name="agnews",
        model_name="textattack/bert-base-uncased-ag-news",
        out_path=os.path.join(output_dir, "gcg_agnews.txt"),
        num_samples=1000,
        num_steps=50,
        adv_string_init="! ! ! ! ! !",
        num_candidates=64,
        topk=256,
    )

    print("\n" + "="*70)
    print("Completed! Generated files:")
    print(f"  - {os.path.join(output_dir, 'gcg_sst2.txt')}")
    print(f"  - {os.path.join(output_dir, 'gcg_agnews.txt')}")
    print("="*70)
