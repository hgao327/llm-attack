import os
import sys

# Must set environment variables before importing torch to disable MPS
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



# ============ LLM-Attacks GCG Implementation ============
def token_gradients(model, input_ids, attention_mask, target_label):
    """
    Compute gradients for each token position
    Based on LLM-Attacks implementation
    """
    # Get embedding layer
    embed_layer = model.get_input_embeddings()

    # Get one-hot encoding
    one_hot = torch.zeros(
        input_ids.shape[0],
        input_ids.shape[1],
        embed_layer.weight.shape[0],
        dtype=torch.float32,
        device=DEVICE
    )
    one_hot.scatter_(2, input_ids.unsqueeze(2), 1.0)
    one_hot.requires_grad_(True)

    # Get embeddings through one-hot
    embeds = torch.matmul(one_hot, embed_layer.weight)

    # Forward pass
    logits = model(inputs_embeds=embeds, attention_mask=attention_mask).logits

    # Compute loss (we want to maximize loss to flip prediction)
    loss = F.cross_entropy(logits, target_label)

    # Backward pass
    loss.backward()

    # Return gradients
    return one_hot.grad.clone()


def sample_control(control_toks, grad, num_candidates, topk=256):
    """
    Sample candidate tokens based on gradients
    Implements LLM-Attacks GCG sampling strategy

    Goal: Maximize loss (increase model's loss on true label)
    Gradient direction: grad points toward loss increase
    """
    if len(grad.shape) == 3:
        # [batch_size, seq_len, vocab_size] -> [seq_len, vocab_size]
        grad = grad.mean(dim=0)

    # Get top-k optimal replacements for each position
    # Select tokens with largest gradients (along positive gradient direction = maximize loss)
    top_indices = grad.topk(topk, dim=-1).indices  # [seq_len, topk]

    # Generate candidates
    control_toks_repeated = control_toks.repeat(num_candidates, 1)

    for i in range(num_candidates):
        # Randomly select a position to modify
        pos = np.random.randint(0, control_toks.shape[1])
        # Randomly select a token from top-k at this position
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
    Generate adversarial samples using GCG (Greedy Coordinate Gradient) method
    Based on "Universal and Transferable Adversarial Attacks on Aligned Language Models"

    Args:
        model: Target model
        tokenizer: Tokenizer
        text: Original text
        label: True label
        num_steps: GCG optimization steps
        adv_string_init: Adversarial string initialization
        num_candidates: Number of candidates generated per step
        topk: Top-k tokens considered for each position
        position: Adversarial string position ('suffix' or 'prefix')
    """
    model.eval()
    model = model.cpu()

    # Original prediction
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.cpu() for k, v in inputs.items()}
        orig_logits = model(**inputs).logits
        orig_pred = orig_logits.argmax(dim=-1).item()

    # If original prediction is already wrong, return directly
    if orig_pred != label:
        return text

    # Initialize adversarial string
    adv_tokens = tokenizer(adv_string_init, add_special_tokens=False, return_tensors="pt")["input_ids"]
    adv_tokens = adv_tokens.cpu()

    # Tokenize original text
    text_tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    text_tokens = text_tokens.cpu()

    best_adv_tokens = adv_tokens.clone()
    best_loss = float('-inf')

    for step in range(num_steps):
        # Build complete input
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

        # Compute gradients
        grad = token_gradients(model, input_ids, attention_mask, target_label)

        # Only take gradients of adversarial tokens
        adv_grad = grad[:, adv_slice, :]

        # Sample candidates
        adv_token_candidates = sample_control(
            adv_tokens,
            adv_grad,
            num_candidates,
            topk=topk
        )

        # Evaluate all candidates
        best_candidate = adv_tokens.clone()
        best_candidate_loss = best_loss

        for candidate in adv_token_candidates:
            # Build complete input
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

            # Evaluate
            with torch.no_grad():
                logits = model(
                    input_ids=candidate_input_ids,
                    attention_mask=candidate_attention_mask
                ).logits
                loss = F.cross_entropy(logits, target_label).item()
                pred = logits.argmax(dim=-1).item()

            # Check if prediction successfully flipped
            if pred != label:
                # Success! Return result
                adv_string = tokenizer.decode(candidate, skip_special_tokens=True)
                if position == 'suffix':
                    return text + " " + adv_string
                else:
                    return adv_string + " " + text

            # Update best candidate
            if loss > best_candidate_loss:
                best_candidate_loss = loss
                best_candidate = candidate.clone()

        # Update adversarial tokens
        if best_candidate_loss > best_loss:
            best_loss = best_candidate_loss
            adv_tokens = best_candidate.unsqueeze(0)
            best_adv_tokens = best_candidate.clone()

    # If flip unsuccessful, return best attempt
    adv_string = tokenizer.decode(best_adv_tokens, skip_special_tokens=True)
    if position == 'suffix':
        return text + " " + adv_string
    else:
        return adv_string + " " + text


# ============ Dataset Unified Interface ============
def get_dataset(name):
    if name == "sst2":
        ds = load_dataset("glue", "sst2", split="train")
        return [(x["sentence"], x["label"]) for x in ds]

    if name == "agnews":
        ds = load_dataset("ag_news", split="train")
        return [(x["text"], x["label"]) for x in ds]

    raise ValueError(name)


# ============ Main Generation Logic ============
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
    Generate adversarial sample dataset using GCG method

    Args:
        dataset_name: Dataset name
        model_name: Model name
        out_path: Output path
        num_samples: Number of samples to generate
        num_steps: GCG optimization steps
        adv_string_init: Adversarial string initialization
        num_candidates: Number of candidates per step
        topk: Top-k sampling
    """
    # Label mapping
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
    # Only take needed number of samples
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

        # Check if successful
        with torch.no_grad():
            inputs = tokenizer(adv_text, return_tensors="pt", truncation=True)
            inputs = {k: v.cpu() for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred = logits.argmax(dim=-1).item()

        if pred != label:
            success_count += 1

        # Write in new format
        label_name = label_names.get(label, f"label_{label}")
        f.write(f"Sample {cnt + 1}:\n")
        f.write(f"    Original: {text.replace(chr(10), ' ')}\n")
        f.write(f"    Adversarial: {adv_text.replace(chr(10), ' ')}\n")
        f.write(f"    Label: {label_name}\n")
        f.write("\n")

        cnt += 1

        # Flush every 10 samples to avoid data loss on program interruption
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

    # SST-2 dataset
    print("\n[1/2] Generating SST-2 adversarial samples...")
    generate_file(
        dataset_name="sst2",
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        out_path=os.path.join(output_dir, "gcg_sst2.txt"),
        num_samples=1000,
        num_steps=50,                # GCG optimization steps
        adv_string_init="! ! ! ! ! !",  # Initial adversarial string (6 tokens)
        num_candidates=64,           # Candidates per step (reduce for speed)
        topk=256,                    # top-k sampling
    )

    # AG News dataset
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
