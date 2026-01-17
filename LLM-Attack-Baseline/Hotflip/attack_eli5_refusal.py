"""
Attack LLMs on ELI5 dataset - Refusal Task
Measure ASR: If LLM refuses to answer after perturbation = attack success
"""

import json
import os
import sys
import re
import random
import numpy as np
from collections import defaultdict

# ========== NumPy compatibility patches ==========
if not hasattr(np, '_core'):
    import types
    _core = types.ModuleType('numpy._core')
    _core.multiarray = np.core.multiarray if hasattr(np.core, 'multiarray') else np
    _core.umath = np.core.umath if hasattr(np.core, 'umath') else np
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = _core.multiarray
    sys.modules['numpy._core.umath'] = _core.umath
    np._core = _core

if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'bool'):
    np.bool = np.bool_
# ========== End NumPy patches ==========

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not installed. Run: pip install transformers torch")
    HAS_TRANSFORMERS = False

def load_word_vectors(w2v_path=None):
    """Load word vectors for similarity-based replacement"""
    import pickle
    
    for pfile in ['sst2.p', 'agnews.p']:
        if os.path.exists(pfile):
            print(f"Loading word vectors from {pfile}...")
            x = None
            for encoding in ['latin1', 'bytes', None]:
                try:
                    with open(pfile, 'rb') as f:
                        if encoding:
                            x = pickle.load(f, encoding=encoding)
                        else:
                            x = pickle.load(f)
                    break
                except Exception as e:
                    continue
            
            if x is None:
                continue
            
            W = x[1]
            word_idx_map = x[3]
            idx_word_map = {v: k for k, v in word_idx_map.items()}
            return W, word_idx_map, idx_word_map
    
    print("Warning: No word vectors found. Using random perturbations.")
    return None, None, None

def find_similar_words(word, W, word_idx_map, idx_word_map, top_k=10, threshold=0.5):
    """Find similar words based on cosine similarity"""
    if word not in word_idx_map:
        return []
    
    word_idx = word_idx_map[word]
    word_vec = W[word_idx]
    
    similarities = []
    for idx in range(len(W)):
        if idx == word_idx or idx == 0:
            continue
        other_vec = W[idx]
        dot = np.dot(word_vec, other_vec)
        norm1 = np.linalg.norm(word_vec)
        norm2 = np.linalg.norm(other_vec)
        if norm1 > 0 and norm2 > 0:
            sim = dot / (norm1 * norm2)
            if sim > threshold:
                similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    similar_words = []
    for idx, sim in similarities[:top_k]:
        if idx in idx_word_map:
            similar_words.append(idx_word_map[idx])
    
    return similar_words

def perturb_text(text, W, word_idx_map, idx_word_map, num_perturbations=2):
    """Perturb text by replacing words with similar words"""
    words = text.lower().split()
    perturbed_words = words.copy()
    
    replaceable = []
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^a-z]', '', word)
        if clean_word in word_idx_map:
            similar = find_similar_words(clean_word, W, word_idx_map, idx_word_map, top_k=5)
            if similar:
                replaceable.append((i, word, clean_word, similar))
    
    if not replaceable:
        return text, []
    
    changes = []
    random.shuffle(replaceable)
    for i, (idx, orig_word, clean_word, similar_words) in enumerate(replaceable[:num_perturbations]):
        new_word = random.choice(similar_words)
        # Preserve punctuation
        if orig_word != clean_word:
            suffix = orig_word[len(clean_word):]
            new_word = new_word + suffix
        perturbed_words[idx] = new_word
        changes.append((clean_word, new_word.rstrip('?.,!')))
    
    return ' '.join(perturbed_words), changes

def load_eli5_data(filepath, max_samples=100):
    """Load ELI5 dataset - handles both JSON array and JSON Lines format"""
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse as JSON array first
    try:
        items = json.loads(content)
        if isinstance(items, list):
            print(f"    Loaded as JSON array with {len(items)} items")
        else:
            items = [items]  # Single object
    except json.JSONDecodeError:
        # Try JSON Lines format
        items = []
        for line in content.strip().split('\n'):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except:
                    continue
        print(f"    Loaded as JSON Lines with {len(items)} items")
    
    # Process items
    for i, item in enumerate(items):
        if i >= max_samples:
            break
        
        # Combine title and selftext as the question
        question = item.get('title', '')
        selftext = item.get('selftext', '')
        if selftext:
            question = question + " " + selftext
        
        # Get the best answer (highest score)
        answers = item.get('answers', {})
        scores = answers.get('score', [])
        texts = answers.get('text', [])
        
        best_answer = ""
        if scores and texts:
            best_idx = scores.index(max(scores))
            best_answer = texts[best_idx] if best_idx < len(texts) else ""
        
        data.append({
            'q_id': item.get('q_id', ''),
            'question': question.strip(),
            'answer': best_answer,
            'subreddit': item.get('subreddit', '')
        })
    
    return data

def generate_with_llm(model, tokenizer, prompt, max_new_tokens=150):
    """Generate text with LLM"""
    # Format as a question-answering prompt
    formatted_prompt = f"Question: {prompt}\n\nAnswer:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    if "Answer:" in generated:
        generated = generated.split("Answer:")[-1].strip()
    return generated

def is_refusal(text):
    """Check if the response is a refusal to answer"""
    text_lower = text.lower().strip()
    
    # Empty or very short response
    if len(text_lower) < 10:
        return True
    
    # Common refusal patterns
    refusal_patterns = [
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
        "i don't understand",
        "i do not understand",
        "sorry, i",
        "i apologize",
        "i'm sorry",
        "i am sorry",
        "unable to answer",
        "cannot answer",
        "can't answer",
        "don't have enough",
        "do not have enough",
        "not enough information",
        "unclear",
        "doesn't make sense",
        "does not make sense",
        "no answer",
        "cannot provide",
        "can't provide",
        "i'm afraid",
        "i am afraid",
        "beyond my",
        "outside my",
        "not qualified",
        "no idea",
        "don't have an answer",
        "do not have an answer",
        "refuse to",
        "won't answer",
        "will not answer",
        "inappropriate",
        "not appropriate",
        "can not help",
        "cannot help",
        "无法回答",
        "不知道",
        "抱歉",
        "对不起",
    ]
    
    for pattern in refusal_patterns:
        if pattern in text_lower:
            return True
    
    # Check if response is just repeating the question
    if text_lower.startswith("question:"):
        return True
    
    return False

def main():
    # Configuration
    ELI5_TEST = "/home/liubingshan/datasets/eli5_test.json"
    ELI5_TRAIN = "/home/liubingshan/datasets/eli5_train.json"
    MAX_SAMPLES = 98
    NUM_PERTURBATIONS = 2
    
    # LLM models
    LLM_MODELS = [
        "/home/llms/Qwen1.5-1.8B-Chat",
        "/home/llms/llama-3.2-3B-instruct",
        "/home/llms/Mistral-7B-v0.1",
        "/home/llms/deepseek-llm-7b-chat"
    ]
    
    print("="*60)
    print("HotFlip-style Attack on LLMs - ELI5 Refusal Task")
    print("="*60)
    
    # Load ELI5 data
    print(f"\n[1] Loading ELI5 data...")
    eli5_data = []
    
    if os.path.exists(ELI5_TEST):
        print(f"    Loading from {ELI5_TEST}...")
        eli5_data = load_eli5_data(ELI5_TEST, max_samples=MAX_SAMPLES)
    elif os.path.exists(ELI5_TRAIN):
        print(f"    Test file not found, loading from {ELI5_TRAIN}...")
        eli5_data = load_eli5_data(ELI5_TRAIN, max_samples=MAX_SAMPLES)
    else:
        print(f"Error: ELI5 files not found!")
        return
    
    print(f"    Loaded {len(eli5_data)} samples")
    
    # Show sample
    if eli5_data:
        print(f"\n    Sample question: {eli5_data[0]['question'][:100]}...")
    
    # Load word vectors
    print("\n[2] Loading word vectors...")
    W, word_idx_map, idx_word_map = load_word_vectors()
    
    if W is not None:
        print(f"    Loaded {len(word_idx_map)} words")
    
    # Test each LLM
    results = {}
    
    for model_path in LLM_MODELS:
        model_name = os.path.basename(model_path)
        print(f"\n{'='*60}")
        print(f"[3] Testing model: {model_name}")
        print("="*60)
        
        if not os.path.exists(model_path):
            print(f"    Model not found: {model_path}")
            continue
        
        if not HAS_TRANSFORMERS:
            print("    Transformers not installed, skipping")
            continue
        
        # Load model
        print(f"    Loading model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"    Failed to load model: {e}")
            continue
        
        # Attack stats
        total = 0
        refusal_original = 0
        refusal_perturbed = 0
        attack_success = 0  # Originally answered, but refused after perturbation
        
        print(f"    Running attack on {len(eli5_data)} samples...")
        
        for i, item in enumerate(eli5_data):
            question = item['question']
            
            # Generate with original question
            try:
                orig_output = generate_with_llm(model, tokenizer, question)
                orig_refused = is_refusal(orig_output)
            except Exception as e:
                print(f"    Sample {i} error: {e}")
                continue
            
            # Perturb question
            if W is not None:
                perturbed_q, changes = perturb_text(
                    question, W, word_idx_map, idx_word_map,
                    num_perturbations=NUM_PERTURBATIONS
                )
            else:
                perturbed_q = question
                changes = []
            
            # Generate with perturbed question
            try:
                pert_output = generate_with_llm(model, tokenizer, perturbed_q)
                pert_refused = is_refusal(pert_output)
            except Exception as e:
                continue
            
            total += 1
            if orig_refused:
                refusal_original += 1
            if pert_refused:
                refusal_perturbed += 1
            
            # Attack success: originally answered, now refused
            if not orig_refused and pert_refused:
                attack_success += 1
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(eli5_data)} samples")
            
            # Print examples
            if i < 2 or (not orig_refused and pert_refused):
                print(f"\n    --- Sample {i+1} ---")
                print(f"    Question: {question[:80]}...")
                print(f"    Changes: {changes}")
                print(f"    Original response: {orig_output[:100]}...")
                print(f"    Original refused: {orig_refused}")
                print(f"    Perturbed response: {pert_output[:100]}...")
                print(f"    Perturbed refused: {pert_refused}")
                if not orig_refused and pert_refused:
                    print(f"    >>> ATTACK SUCCESS! <<<")
        
        # Calculate rates
        if total > 0:
            refusal_rate_orig = refusal_original / total * 100
            refusal_rate_pert = refusal_perturbed / total * 100
            asr = attack_success / total * 100
            # ASR among answerable questions
            answerable = total - refusal_original
            if answerable > 0:
                asr_answerable = attack_success / answerable * 100
            else:
                asr_answerable = 0
        else:
            refusal_rate_orig = refusal_rate_pert = asr = asr_answerable = 0
        
        results[model_name] = {
            'total': total,
            'refusal_original': refusal_rate_orig,
            'refusal_perturbed': refusal_rate_pert,
            'asr': asr,
            'asr_answerable': asr_answerable,
            'attack_success_count': attack_success
        }
        
        print(f"\n    Results for {model_name}:")
        print(f"    - Total samples: {total}")
        print(f"    - Original refusal rate: {refusal_rate_orig:.2f}%")
        print(f"    - Perturbed refusal rate: {refusal_rate_pert:.2f}%")
        print(f"    - ASR (attack success rate): {asr:.2f}%")
        print(f"    - ASR among answerable: {asr_answerable:.2f}%")
        
        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - ELI5 Refusal Attack")
    print("="*70)
    print(f"{'Model':<25} {'Orig Refusal':<14} {'Pert Refusal':<14} {'ASR':<10} {'ASR(ans)':<10}")
    print("-"*70)
    for model_name, stats in results.items():
        print(f"{model_name:<25} {stats['refusal_original']:>11.2f}% {stats['refusal_perturbed']:>11.2f}% {stats['asr']:>7.2f}% {stats['asr_answerable']:>7.2f}%")
    
    # Save results
    output_file = "eli5_refusal_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

