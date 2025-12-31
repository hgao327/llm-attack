import os

# Must set environment variables before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# Force CPU to avoid MPS bus errors
DEVICE = "cpu"
torch.set_default_device('cpu')
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False


# ============ AutoPrompt Attack Implementation ============

def encode_label(tokenizer, label_text):
    """Encode label text to token id"""
    # For BERT tokenizer, ensure we return only one token
    tokens = tokenizer.tokenize(label_text)
    if len(tokens) == 0:
        raise ValueError(f"Label '{label_text}' tokenizes to empty")
    # Use only the first token
    token_id = tokenizer.convert_tokens_to_ids(tokens[0])
    return token_id


def hotflip_attack(averaged_grad, embedding_matrix, current_embedding, num_candidates=10):
    """
    HotFlip: gradient-based optimal token replacement

    Args:
        averaged_grad: [embedding_dim] gradient vector
        embedding_matrix: [vocab_size, embedding_dim] word embedding matrix
        current_embedding: [embedding_dim] current token embedding
        num_candidates: number of candidates to return

    Returns:
        top_k_ids: top k optimal token ids
    """
    with torch.no_grad():
        # Standard HotFlip: score = (E - e_current) · grad
        # This computes first-order approximation of loss change after replacement
        gradient_dot_embedding = torch.matmul(
            embedding_matrix - current_embedding.unsqueeze(0),
            averaged_grad
        )

        # Get top-k candidates (higher score = larger loss change)
        _, top_k_ids = gradient_dot_embedding.topk(num_candidates)

    return top_k_ids


def autoprompt_attack(
    model,
    tokenizer,
    text,
    true_label,
    label_map,
    num_trigger_tokens=3,
    num_candidates=40,
    num_iterations=10,
):
    """
    Adversarial AutoPrompt: Use MLM and cloze prompting to generate adversarial samples

    Template: [T] [T] [T] {sentence} [MASK]
    Goal: Optimize [T] tokens to flip [MASK] prediction to wrong label token

    Args:
        model: MLM model
        tokenizer: Tokenizer
        text: Original text
        true_label: True label (0 or 1)
        label_map: {0: "bad", 1: "good"} - verbalizer mapping
        num_trigger_tokens: Number of trigger tokens
        num_candidates: Number of candidates per iteration
        num_iterations: Number of optimization iterations

    Returns:
        Adversarial sample text
    """
    model.eval()
    embeddings = model.get_input_embeddings()

    # Label tokens - only compare among these tokens
    all_label_ids = [encode_label(tokenizer, label_text) for label_text in label_map.values()]
    true_label_token_id = all_label_ids[true_label]

    # Initialize trigger tokens (random common words)
    trigger_token_ids = [
        tokenizer.convert_tokens_to_ids(random.choice(['the', 'a', 'is', 'was', 'are']))
        for _ in range(num_trigger_tokens)
    ]
    trigger_token_ids = torch.tensor(trigger_token_ids, device=DEVICE)

    # Tokenize original text
    text_encoding = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    text_token_ids = text_encoding["input_ids"][0].to(DEVICE)

    best_trigger_ids = trigger_token_ids.clone()
    best_loss = float('-inf')  # Adversarial goal: maximize loss

    for iteration in range(num_iterations):
        # Build input: [CLS] [T] [T] [T] text [MASK] [SEP]
        input_ids = torch.cat([
            torch.tensor([tokenizer.cls_token_id], device=DEVICE),
            trigger_token_ids,
            text_token_ids,
            torch.tensor([tokenizer.mask_token_id], device=DEVICE),
            torch.tensor([tokenizer.sep_token_id], device=DEVICE)
        ]).unsqueeze(0)

        # Trigger positions: [1, 2, ..., num_trigger_tokens]
        trigger_positions = list(range(1, 1 + num_trigger_tokens))
        # MASK position
        mask_position = 1 + num_trigger_tokens + len(text_token_ids)

        attention_mask = torch.ones_like(input_ids)

        # Forward + backward pass
        model.zero_grad()
        embeds = embeddings(input_ids)
        embeds.requires_grad_(True)
        embeds.retain_grad()

        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Logits at [MASK] position
        mask_logits = logits[0, mask_position, :]  # [vocab_size]

        # Adversarial objective: maximize loss (reduce true label probability)
        loss = -F.log_softmax(mask_logits, dim=-1)[true_label_token_id]
        loss.backward()

        # Get gradients for trigger tokens
        trigger_grad = embeds.grad[0, trigger_positions, :]  # [num_trigger_tokens, embedding_dim]

        # Check current prediction (compare only among label tokens)
        label_logits = torch.stack([mask_logits[lid] for lid in all_label_ids])
        predicted_label_idx = label_logits.argmax().item()
        current_loss = loss.item()

        # Maintain trigger with maximum loss
        if current_loss > best_loss:
            best_loss = current_loss
            best_trigger_ids = trigger_token_ids.clone()

        # Use HotFlip to update one trigger token
        token_to_flip = random.randint(0, num_trigger_tokens - 1)

        # Get current token embedding
        current_token_id = trigger_token_ids[token_to_flip]
        current_embedding = embeddings.weight[current_token_id]

        # Get candidates
        candidates = hotflip_attack(
            trigger_grad[token_to_flip],
            embeddings.weight,
            current_embedding,
            num_candidates=num_candidates
        )

        # Evaluate candidates
        best_candidate_loss = current_loss
        best_candidate = trigger_token_ids[token_to_flip].item()

        for candidate in candidates:
            # Create temporary trigger
            temp_trigger = trigger_token_ids.clone()
            temp_trigger[token_to_flip] = candidate

            # Build input
            temp_input_ids = torch.cat([
                torch.tensor([tokenizer.cls_token_id], device=DEVICE),
                temp_trigger,
                text_token_ids,
                torch.tensor([tokenizer.mask_token_id], device=DEVICE),
                torch.tensor([tokenizer.sep_token_id], device=DEVICE)
            ]).unsqueeze(0)

            temp_attention_mask = torch.ones_like(temp_input_ids)

            # Evaluate
            with torch.no_grad():
                temp_outputs = model(input_ids=temp_input_ids, attention_mask=temp_attention_mask)
                temp_mask_logits = temp_outputs.logits[0, mask_position, :]

                # Compare only among label tokens
                temp_label_logits = torch.stack([temp_mask_logits[lid] for lid in all_label_ids])
                temp_pred_idx = temp_label_logits.argmax().item()
                temp_loss = -F.log_softmax(temp_mask_logits, dim=-1)[true_label_token_id].item()

            # If successfully flipped and loss is higher (early stopping)
            if temp_pred_idx != true_label and temp_loss > best_loss:
                best_loss = temp_loss
                best_trigger_ids = temp_trigger.clone()
                trigger_token_ids = temp_trigger
                break

            # Select candidate with highest loss
            if temp_loss > best_candidate_loss:
                best_candidate_loss = temp_loss
                best_candidate = candidate.item()

        # Update trigger token
        trigger_token_ids[token_to_flip] = best_candidate

        # Update best if better candidate found
        if best_candidate_loss > best_loss:
            best_loss = best_candidate_loss
            best_trigger_ids = trigger_token_ids.clone()

    # Generate adversarial sample (return trigger ids and text)
    return best_trigger_ids, text_token_ids


def get_dataset(name):
    """Load dataset"""
    if name == "sst2":
        ds = load_dataset("glue", "sst2", split="train")
        return [(x["sentence"], x["label"]) for x in ds]

    if name == "agnews":
        ds = load_dataset("ag_news", split="train")
        return [(x["text"], x["label"]) for x in ds]

    raise ValueError(name)


def predict_with_autoprompt(model, tokenizer, trigger_ids, text_ids, label_map):
    """
    Predict using AutoPrompt method

    Template: [CLS] {triggers} {text} [MASK] [SEP]
    Prediction: Compare logits only among label tokens

    Args:
        trigger_ids: trigger token ids (tensor)
        text_ids: text token ids (tensor)
        label_map: {0: "bad", 1: "good"}

    Returns:
        predicted label index (int)
    """
    model.eval()

    # Build input
    input_ids = torch.cat([
        torch.tensor([tokenizer.cls_token_id], device=DEVICE),
        trigger_ids,
        text_ids,
        torch.tensor([tokenizer.mask_token_id], device=DEVICE),
        torch.tensor([tokenizer.sep_token_id], device=DEVICE)
    ]).unsqueeze(0)

    mask_position = len(input_ids[0]) - 2  # [MASK] before [SEP]
    attention_mask = torch.ones_like(input_ids)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        mask_logits = outputs.logits[0, mask_position, :]  # [vocab_size]

    # Compare only among label tokens
    all_label_ids = [encode_label(tokenizer, label_text) for label_text in label_map.values()]
    label_logits = torch.stack([mask_logits[lid] for lid in all_label_ids])

    # Predict label (argmax over label tokens only)
    pred_label = label_logits.argmax().item()
    return pred_label


def generate_file_autoprompt(
    dataset_name,
    model_name,
    out_path,
    num_samples=1000,
    num_trigger_tokens=3,
    num_iterations=10,
):
    """
    Generate adversarial samples using AutoPrompt method

    Args:
        dataset_name: Dataset name ('sst2' or 'agnews')
        model_name: MLM model name
        out_path: Output file path
        num_samples: Number of samples to generate
        num_trigger_tokens: Number of trigger tokens
        num_iterations: Number of optimization iterations
    """
    # Label mapping - use single-token verbalizers
    if dataset_name == "sst2":
        label_names = {0: "negative", 1: "positive"}
        label_map = {0: "bad", 1: "good"}  # Single-token verbalizers
    elif dataset_name == "agnews":
        label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        # Ensure all are single tokens
        label_map = {0: "world", 1: "sports", 2: "business", 3: "tech"}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"\nLoading MLM model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)

    # Verify all label tokens are single tokens
    print("Verifying label tokens:")
    for idx, label_text in label_map.items():
        tokens = tokenizer.tokenize(label_text)
        if len(tokens) != 1:
            raise ValueError(f"Label '{label_text}' is not a single token: {tokens}")
        print(f"  {idx} -> '{label_text}' (token: {tokens[0]})")

    print(f"\nLoading dataset: {dataset_name}")
    data = get_dataset(dataset_name)
    random.shuffle(data)
    # Take only the needed number of samples
    data = data[:num_samples]

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    f = open(out_path, "w", encoding="utf-8")

    success_count = 0
    cnt = 0

    for text, label in tqdm(data, desc=f"Generating {dataset_name} with AutoPrompt", total=num_samples):
        if cnt >= num_samples:
            break

        # Generate adversarial sample (returns token ids)
        trigger_ids, text_ids = autoprompt_attack(
            model,
            tokenizer,
            text,
            label,
            label_map,
            num_trigger_tokens=num_trigger_tokens,
            num_candidates=40,
            num_iterations=num_iterations,
        )

        # Predict using AutoPrompt method
        pred_label = predict_with_autoprompt(model, tokenizer, trigger_ids, text_ids, label_map)

        if pred_label != label:
            success_count += 1

        # Build adversarial text (for display)
        trigger_text = tokenizer.decode(trigger_ids, skip_special_tokens=True)
        adv_text = trigger_text + " " + text

        # Write in new format
        label_name = label_names.get(label, f"label_{label}")
        f.write(f"Sample {cnt + 1}:\n")
        f.write(f"    Original: {text.replace(chr(10), ' ')}\n")
        f.write(f"    Adversarial: {adv_text.replace(chr(10), ' ')}\n")
        f.write(f"    Label: {label_name}\n")
        f.write("\n")

        cnt += 1

        # Flush every 10 samples to avoid data loss on interruption
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
    import platform
    print(platform.platform())
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")

    print("\n" + "="*70)
    print("AutoPrompt Attack (MLM + Cloze Prompting)")
    print("="*70)

    # Create output dir (in parent)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # SST-2 dataset
    print("\n[1/2] Generating SST-2 adversarial samples...")
    generate_file_autoprompt(
        dataset_name="sst2",
        model_name="bert-base-uncased",  # Use BERT MLM
        out_path=os.path.join(output_dir, "autoprompt_sst2.txt"),
        num_samples=1000,
        num_trigger_tokens=3,
        num_iterations=10,
    )

    # AG News dataset
    print("\n[2/2] Generating AG News adversarial samples...")
    generate_file_autoprompt(
        dataset_name="agnews",
        model_name="bert-base-uncased",
        out_path=os.path.join(output_dir, "autoprompt_agnews.txt"),
        num_samples=1000,
        num_trigger_tokens=3,
        num_iterations=10,
    )

    print("\n" + "="*70)
    print("Completed! Generated files:")
    print(f"  - {os.path.join(output_dir, 'autoprompt_sst2.txt')}")
    print(f"  - {os.path.join(output_dir, 'autoprompt_agnews.txt')}")
    print("="*70)
