#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 JSONL 格式的 SST2 数据转换为代码需要的 pickle 格式
用法: python convert_jsonl_to_pickle.py

输入格式（JSONL）:
{"text": "no movement , no yuks , not much of anything .", "label": 0, "label_text": "negative"}

输出: sst2_new.p (pickle 文件)

重要：此脚本会保留原有 sst2.p 中的 word_idx_map 和词向量，
只更新数据部分，确保与预训练模型兼容！
"""

import json
import pickle
import numpy as np
import re
import os
import sys

# 添加 numpy 兼容性
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core

# 数据路径 - 修改为你的实际路径
TRAIN_FILE = "/home/liubingshan/datasets/SST2_train.jsonl"
TEST_FILE = "/home/liubingshan/datasets/SST2_test.jsonl"
ORIGINAL_PICKLE = "sst2.p"  # 原始 pickle 文件（保留 word_idx_map）
OUTPUT_FILE = "sst2_new.p"  # 新的输出文件

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_jsonl(filepath):
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def build_data(train_data, test_data, clean_string=True):
    """
    Build dataset in the format expected by the model
    """
    revs = []
    vocab = {}
    max_l = 0
    
    # Process training data
    for item in train_data:
        text = item['text']
        label = item['label']
        if clean_string:
            orig_rev = clean_str(text)
        else:
            orig_rev = text.lower()
        words = orig_rev.split()
        max_l = max(max_l, len(words))
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # 0 is reserved for padding
        datum = {
            "y": label,
            "text": orig_rev,
            "num_words": len(words),
            "split": 0  # 0 for training
        }
        revs.append(datum)
    
    # Process test data
    for item in test_data:
        text = item['text']
        label = item['label']
        if clean_string:
            orig_rev = clean_str(text)
        else:
            orig_rev = text.lower()
        words = orig_rev.split()
        max_l = max(max_l, len(words))
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab) + 1
        datum = {
            "y": label,
            "text": orig_rev,
            "num_words": len(words),
            "split": 1  # 1 for test
        }
        revs.append(datum)
    
    return revs, vocab, max_l

def load_word2vec(fname, vocab):
    """
    Load word2vec vectors for words in vocab
    """
    from gensim.models import KeyedVectors
    
    print("Loading word2vec...")
    word_vecs = KeyedVectors.load_word2vec_format(fname, binary=True)
    
    k = 300  # word2vec dimension
    W = np.zeros(shape=(len(vocab)+1, k), dtype='float32')
    W2 = np.random.uniform(-0.25, 0.25, (len(vocab)+1, k)).astype('float32')
    
    word_idx_map = {}
    word_idx_map_invert = {}
    
    found = 0
    for word, idx in vocab.items():
        word_idx_map[word] = idx
        word_idx_map_invert[idx] = word
        if word in word_vecs:
            W[idx] = word_vecs[word]
            W2[idx] = word_vecs[word]
            found += 1
        else:
            W2[idx] = np.random.uniform(-0.25, 0.25, k)
    
    print(f"Found {found}/{len(vocab)} words in word2vec")
    return W, W2, word_idx_map

def build_data_with_existing_vocab(train_data, test_data, word_idx_map, clean_string=True):
    """
    Build dataset using existing word_idx_map (for compatibility with pretrained model)
    """
    revs = []
    max_l = 0
    unknown_words = set()
    
    # Process training data
    for item in train_data:
        text = item['text']
        label = item['label']
        if clean_string:
            orig_rev = clean_str(text)
        else:
            orig_rev = text.lower()
        words = orig_rev.split()
        max_l = max(max_l, len(words))
        
        # Check for unknown words
        for word in words:
            if word not in word_idx_map:
                unknown_words.add(word)
        
        datum = {
            "y": label,
            "text": orig_rev,
            "num_words": len(words),
            "split": 0  # 0 for training
        }
        revs.append(datum)
    
    # Process test data
    for item in test_data:
        text = item['text']
        label = item['label']
        if clean_string:
            orig_rev = clean_str(text)
        else:
            orig_rev = text.lower()
        words = orig_rev.split()
        max_l = max(max_l, len(words))
        
        for word in words:
            if word not in word_idx_map:
                unknown_words.add(word)
        
        datum = {
            "y": label,
            "text": orig_rev,
            "num_words": len(words),
            "split": 1  # 1 for test
        }
        revs.append(datum)
    
    if unknown_words:
        print(f"Warning: {len(unknown_words)} unknown words (will use random vectors)")
    
    return revs, max_l

def main():
    # Load original pickle to get word_idx_map, W, W2
    print(f"Loading original pickle from {ORIGINAL_PICKLE}...")
    try:
        with open(ORIGINAL_PICKLE, 'rb') as f:
            x = pickle.load(f, encoding='latin1')
        _, W, W2, word_idx_map, vocab, _ = x[0], x[1], x[2], x[3], x[4], x[5]
        print(f"Loaded existing word_idx_map with {len(word_idx_map)} words")
    except Exception as e:
        print(f"Error loading {ORIGINAL_PICKLE}: {e}")
        print("Please make sure sst2.p exists!")
        return
    
    print(f"\nLoading training data from {TRAIN_FILE}...")
    train_data = load_jsonl(TRAIN_FILE)
    print(f"Loaded {len(train_data)} training samples")
    
    print(f"Loading test data from {TEST_FILE}...")
    test_data = load_jsonl(TEST_FILE)
    print(f"Loaded {len(test_data)} test samples")
    
    print("\nBuilding dataset with existing vocabulary...")
    revs, max_l = build_data_with_existing_vocab(train_data, test_data, word_idx_map)
    
    # Save new pickle with same W, W2, word_idx_map but new revs
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump([revs, W, W2, word_idx_map, vocab, max_l], f)
    
    print("\n" + "="*50)
    print("Done! Statistics:")
    print("="*50)
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Total samples: {len(revs)}")
    print(f"  Vocabulary size: {len(word_idx_map)}")
    print(f"  Max sentence length: {max_l}")
    print(f"\nOutput file: {OUTPUT_FILE}")
    print("\nTo use the new dataset, modify use_pretrained_gene_testset.py:")
    print('  Change: x = pickle.load(open(THEME+".p","rb"))')
    print('  To:     x = pickle.load(open("sst2_new.p","rb"), encoding="latin1")')

if __name__ == "__main__":
    main()

