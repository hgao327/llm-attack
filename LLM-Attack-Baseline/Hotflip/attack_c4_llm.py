"""
Attack LLMs on C4 dataset with word perturbations
Measure ASR (Attack Success Rate) based on sentiment of generated text
"""

import json
import os
import sys
import re
import random
import numpy as np
from collections import defaultdict

# ========== NumPy compatibility patches ==========
# Add numpy._core compatibility for pickle loading
if not hasattr(np, '_core'):
    import types
    _core = types.ModuleType('numpy._core')
    _core.multiarray = np.core.multiarray if hasattr(np.core, 'multiarray') else np
    _core.umath = np.core.umath if hasattr(np.core, 'umath') else np
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = _core.multiarray
    sys.modules['numpy._core.umath'] = _core.umath
    np._core = _core

# Fix deprecated aliases
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'bool'):
    np.bool = np.bool_
# ========== End NumPy patches ==========

# Try to import transformers for LLM inference
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not installed. Run: pip install transformers torch")
    HAS_TRANSFORMERS = False

# Try to import sentiment analysis
try:
    from transformers import pipeline as hf_pipeline
    HAS_SENTIMENT = True
except:
    HAS_SENTIMENT = False

def clean_str(string):
    """Clean text for processing"""
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.\-]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def load_c4_data(filepath, max_samples=100):
    """Load C4 dataset from JSON file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if line.strip():
                item = json.loads(line)
                data.append({
                    'prompt': item['prompt'],
                    'natural_text': item.get('natural_text', '')
                })
    return data

def load_word_vectors(w2v_path=None):
    """Load word vectors for similarity-based replacement"""
    import pickle
    
    # Check if sst2.p or agnews.p exists for word vectors
    for pfile in ['sst2.p', 'agnews.p']:
        if os.path.exists(pfile):
            print(f"Loading word vectors from {pfile}...")
            
            # Try multiple methods to load pickle
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
                    print(f"    Pickle load with encoding={encoding} failed: {e}")
                    continue
            
            if x is None:
                print(f"    Failed to load {pfile}, trying next...")
                continue
            
            # x[1] is W (word2vec), x[3] is word_idx_map
            W = x[1]
            word_idx_map = x[3]
            idx_word_map = {v: k for k, v in word_idx_map.items()}
            return W, word_idx_map, idx_word_map
    
    print("Warning: No word vectors found. Using random perturbations.")
    return None, None, None

def find_similar_words(word, W, word_idx_map, idx_word_map, top_k=10, threshold=0.5):
    """Find similar words based on word vector cosine similarity"""
    if word not in word_idx_map:
        return []
    
    word_idx = word_idx_map[word]
    word_vec = W[word_idx]
    
    # Compute cosine similarity with all words
    similarities = []
    for idx in range(len(W)):
        if idx == word_idx or idx == 0:  # Skip same word and padding
            continue
        other_vec = W[idx]
        # Cosine similarity
        dot = np.dot(word_vec, other_vec)
        norm1 = np.linalg.norm(word_vec)
        norm2 = np.linalg.norm(other_vec)
        if norm1 > 0 and norm2 > 0:
            sim = dot / (norm1 * norm2)
            if sim > threshold:
                similarities.append((idx, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k similar words
    similar_words = []
    for idx, sim in similarities[:top_k]:
        if idx in idx_word_map:
            similar_words.append(idx_word_map[idx])
    
    return similar_words

def perturb_text(text, W, word_idx_map, idx_word_map, num_perturbations=1):
    """Perturb text by replacing words with similar words"""
    words = text.lower().split()
    perturbed_words = words.copy()
    
    # Find words that can be replaced
    replaceable = []
    for i, word in enumerate(words):
        if word in word_idx_map:
            similar = find_similar_words(word, W, word_idx_map, idx_word_map, top_k=5)
            if similar:
                replaceable.append((i, word, similar))
    
    if not replaceable:
        return text, []
    
    # Replace up to num_perturbations words
    changes = []
    random.shuffle(replaceable)
    for i, (idx, orig_word, similar_words) in enumerate(replaceable[:num_perturbations]):
        new_word = random.choice(similar_words)
        perturbed_words[idx] = new_word
        changes.append((orig_word, new_word))
    
    return ' '.join(perturbed_words), changes

def generate_with_llm(model, tokenizer, prompt, max_new_tokens=100):
    """Generate text with LLM"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
    # Remove the prompt from the output
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated.strip()

def analyze_sentiment(text, sentiment_analyzer):
    """Analyze sentiment of text, return True if negative"""
    if sentiment_analyzer is None:
        # Fallback: simple keyword matching
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'negative', 
                         'worst', 'fail', 'hate', 'angry', 'sad', 'disappointed',
                         'wrong', 'error', 'problem', 'issue', 'concern', 'risk',
                         'danger', 'threat', 'loss', 'decline', 'drop', 'fall']
        text_lower = text.lower()
        return any(word in text_lower for word in negative_words)
    
    try:
        result = sentiment_analyzer(text[:512])  # Truncate for model
        if isinstance(result, list):
            result = result[0]
        label = result.get('label', '').lower()
        return 'negative' in label or 'neg' in label
    except:
        return False

def main():
    # Configuration
    C4_PATH = "/home/liubingshan/datasets/processed_c4.json"
    MAX_SAMPLES = 98  # Number of samples to test
    NUM_PERTURBATIONS = 2  # Number of words to replace per prompt
    
    # LLM models to test
    LLM_MODELS = [
        "/home/llms/Qwen1.5-1.8B-Chat",
        "/home/llms/llama-3.2-3B-instruct",
        "/home/llms/Mistral-7B-v0.1",
        "/home/llms/deepseek-llm-7b-chat"
    ]
    
    print("="*60)
    print("HotFlip-style Attack on LLMs with C4 Dataset")
    print("="*60)
    
    # Load C4 data
    print(f"\n[1] Loading C4 data from {C4_PATH}...")
    if not os.path.exists(C4_PATH):
        print(f"Error: C4 file not found at {C4_PATH}")
        return
    
    c4_data = load_c4_data(C4_PATH, max_samples=MAX_SAMPLES)
    print(f"    Loaded {len(c4_data)} samples")
    
    # Load word vectors
    print("\n[2] Loading word vectors...")
    W, word_idx_map, idx_word_map = load_word_vectors()
    
    if W is None:
        print("    No word vectors available, using random perturbations")
    else:
        print(f"    Loaded {len(word_idx_map)} words")
    
    # Initialize sentiment analyzer
    print("\n[3] Initializing sentiment analyzer...")
    sentiment_analyzer = None
    if HAS_SENTIMENT:
        try:
            sentiment_analyzer = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            print("    Using DistilBERT sentiment analyzer")
        except Exception as e:
            print(f"    Failed to load sentiment model: {e}")
            print("    Using keyword-based sentiment detection")
    else:
        print("    Using keyword-based sentiment detection")
    
    # Test each LLM
    results = {}
    
    for model_path in LLM_MODELS:
        model_name = os.path.basename(model_path)
        print(f"\n{'='*60}")
        print(f"[4] Testing model: {model_name}")
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
        success_original = 0  # Negative sentiment on original prompt
        success_perturbed = 0  # Negative sentiment on perturbed prompt
        
        print(f"    Running attack on {len(c4_data)} samples...")
        
        for i, item in enumerate(c4_data):
            prompt = item['prompt']
            
            # Generate with original prompt
            try:
                orig_output = generate_with_llm(model, tokenizer, prompt)
                orig_negative = analyze_sentiment(orig_output, sentiment_analyzer)
            except Exception as e:
                print(f"    Sample {i} error: {e}")
                continue
            
            # Perturb prompt
            if W is not None:
                perturbed_prompt, changes = perturb_text(
                    prompt, W, word_idx_map, idx_word_map, 
                    num_perturbations=NUM_PERTURBATIONS
                )
            else:
                perturbed_prompt = prompt
                changes = []
            
            # Generate with perturbed prompt
            try:
                pert_output = generate_with_llm(model, tokenizer, perturbed_prompt)
                pert_negative = analyze_sentiment(pert_output, sentiment_analyzer)
            except Exception as e:
                continue
            
            total += 1
            if orig_negative:
                success_original += 1
            if pert_negative:
                success_perturbed += 1
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(c4_data)} samples")
            
            # Print example
            if i < 3 or (changes and pert_negative and not orig_negative):
                print(f"\n    --- Sample {i+1} ---")
                print(f"    Original prompt: {prompt[:100]}...")
                print(f"    Changes: {changes}")
                print(f"    Original output sentiment: {'NEGATIVE' if orig_negative else 'POSITIVE'}")
                print(f"    Perturbed output sentiment: {'NEGATIVE' if pert_negative else 'POSITIVE'}")
        
        # Calculate ASR
        if total > 0:
            asr_original = success_original / total * 100
            asr_perturbed = success_perturbed / total * 100
            asr_delta = asr_perturbed - asr_original
        else:
            asr_original = asr_perturbed = asr_delta = 0
        
        results[model_name] = {
            'total': total,
            'asr_original': asr_original,
            'asr_perturbed': asr_perturbed,
            'asr_delta': asr_delta
        }
        
        print(f"\n    Results for {model_name}:")
        print(f"    - Total samples: {total}")
        print(f"    - ASR (original): {asr_original:.2f}%")
        print(f"    - ASR (perturbed): {asr_perturbed:.2f}%")
        print(f"    - ASR increase: {asr_delta:+.2f}%")
        
        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Original ASR':<15} {'Perturbed ASR':<15} {'Delta':<10}")
    print("-"*60)
    for model_name, stats in results.items():
        print(f"{model_name:<30} {stats['asr_original']:>12.2f}% {stats['asr_perturbed']:>13.2f}% {stats['asr_delta']:>+8.2f}%")
    
    # Save results
    output_file = "c4_attack_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

