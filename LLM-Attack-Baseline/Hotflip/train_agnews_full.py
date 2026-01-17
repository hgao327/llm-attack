#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AG News 训练完整脚本
用法: python train_agnews_full.py -nonstatic -word2vec

训练完成后会生成:
- agnews.p (数据文件)
- save/agnews_model.pickle (模型文件)
"""

import numpy as np
import sys
import os
import pickle
import csv
import re
import time
from collections import defaultdict
import warnings

# NumPy 兼容性
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float  
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

import theano
import theano.tensor as T

warnings.filterwarnings("ignore")

# ============ 配置 ============
TRAIN_FILE = "/home/liubingshan/datasets/ag_news/train.csv"
TEST_FILE = "/home/liubingshan/datasets/ag_news/test.csv"
WORD2VEC_FILE = "SentDataPre-master/data/GoogleNews-vectors-negative300.bin"
OUTPUT_PICKLE = "agnews.p"
OUTPUT_MODEL = "save/agnews_model.pickle"

# 减少样本量加速训练
MAX_TRAIN_SAMPLES = 10000  # 只用 20000 个训练样本
MAX_TEST_SAMPLES = 1000    # 只用 5000 个测试样本

# ============ 辅助函数 ============
def ReLU(x):
    return T.maximum(0.0, x)

def Iden(x):
    return x

def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """Adadelta update rule"""
    updates = []
    exp_sqr_grads = []
    exp_sqr_ups = []
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads.append(theano.shared(value=empty.astype(theano.config.floatX), name="exp_grad_%s" % param.name))
        gparams.append(T.grad(cost, param))
        exp_sqr_ups.append(theano.shared(value=empty.astype(theano.config.floatX), name="exp_ups_%s" % param.name))
    
    for param, exp_sqr_grad, gp, exp_sqr_up in zip(params, exp_sqr_grads, gparams, exp_sqr_ups):
        new_exp_sqr_grad = rho * exp_sqr_grad + (1 - rho) * T.sqr(gp)
        updates.append((exp_sqr_grad, new_exp_sqr_grad))
        
        step = -(T.sqrt(exp_sqr_up + epsilon) / T.sqrt(new_exp_sqr_grad + epsilon)) * gp
        new_exp_sqr_up = rho * exp_sqr_up + (1 - rho) * T.sqr(step)
        updates.append((exp_sqr_up, new_exp_sqr_up))
        
        if param.name and param.name != word_vec_name:
            stepped_param = param + step
            if param.get_value(borrow=True).ndim == 2:
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates.append((param, stepped_param * scale))
            else:
                updates.append((param, stepped_param))
        else:
            updates.append((param, param + step))
    
    return updates

def clean_str(string):
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

def load_agnews_data():
    """加载 AG News 数据（限制样本量）"""
    print("Loading AG News data...")
    revs = []
    vocab = defaultdict(int)
    max_l = 0
    train_count = 0
    test_count = 0
    
    # 训练集（限制数量）
    with open(TRAIN_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if train_count >= MAX_TRAIN_SAMPLES:
                break
            if len(row) >= 3:
                label = int(row[0]) - 1  # 1-4 -> 0-3
                text = clean_str(row[1] + " " + row[2])
                words = text.split()
                max_l = max(max_l, len(words))
                for word in words:
                    vocab[word] += 1
                revs.append({"y": label, "text": text, "num_words": len(words), "split": 0})
                train_count += 1
    
    # 测试集（限制数量）
    with open(TEST_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if test_count >= MAX_TEST_SAMPLES:
                break
            if len(row) >= 3:
                label = int(row[0]) - 1
                text = clean_str(row[1] + " " + row[2])
                words = text.split()
                max_l = max(max_l, len(words))
                for word in words:
                    vocab[word] += 1
                revs.append({"y": label, "text": text, "num_words": len(words), "split": 1})
                test_count += 1
    
    print(f"  Train samples: {train_count} (max: {MAX_TRAIN_SAMPLES})")
    print(f"  Test samples: {test_count} (max: {MAX_TEST_SAMPLES})")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Max sentence length: {max_l}")
    return revs, vocab, max_l

def load_word2vec(vocab):
    """加载 word2vec"""
    from gensim.models import KeyedVectors
    print("Loading word2vec (this may take a few minutes)...")
    
    w2v = KeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=True)
    
    k = 300
    word_idx_map = {}
    W = np.zeros((len(vocab)+1, k), dtype='float32')
    W2 = np.random.uniform(-0.25, 0.25, (len(vocab)+1, k)).astype('float32')
    
    found = 0
    for i, word in enumerate(vocab.keys(), 1):
        word_idx_map[word] = i
        if word in w2v:
            W[i] = w2v[word]
            W2[i] = w2v[word]
            found += 1
    
    print(f"  Found {found}/{len(vocab)} words in word2vec")
    return W, W2, word_idx_map

def get_idx_from_sent(sent, word_idx_map, max_l, k=300, filter_h=5):
    x = []
    pad = filter_h - 1
    for _ in range(pad):
        x.append(0)
    for word in sent.split():
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2*pad:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l, k=300, filter_h=5):
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"] == 0:
            train.append(sent)
        else:
            test.append(sent)
    return np.array(train, dtype="int"), np.array(test, dtype="int")

def shared_dataset(data_x, data_y):
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(data_y, dtype='int32'), borrow=True)
    return shared_x, shared_y

# ============ 主程序 ============
if __name__ == "__main__":
    print("="*60)
    print("AG News CNN 分类器训练")
    print("="*60)
    
    # 1. 加载数据
    revs, vocab, max_l = load_agnews_data()
    
    # 2. 加载 word2vec
    W, W2, word_idx_map = load_word2vec(vocab)
    
    # 3. 保存数据 pickle
    print(f"\nSaving data to {OUTPUT_PICKLE}...")
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump([revs, W, W2, word_idx_map, vocab, max_l], f)
    
    # 4. 准备训练
    mode = sys.argv[1] if len(sys.argv) > 1 else "-nonstatic"
    word_vectors = sys.argv[2] if len(sys.argv) > 2 else "-word2vec"
    
    non_static = (mode == "-nonstatic")
    U = W if word_vectors == "-word2vec" else W2
    
    print(f"\nModel: CNN-{'non-static' if non_static else 'static'}")
    print(f"Vectors: {'word2vec' if word_vectors == '-word2vec' else 'random'}")
    
    # 加载 CNN 类
    exec(open("conv_net_classes.py").read())
    
    # 准备数据
    train_data, test_data = make_idx_data(revs, word_idx_map, max_l)
    
    # 训练参数
    img_h = train_data.shape[1] - 1
    img_w = 300
    filter_hs = [3, 4, 5]
    hidden_units = [100, 4]  # 4 classes!
    dropout_rate = [0.5]
    batch_size = 50
    n_epochs = 10  # 减少到10个epoch加速训练
    lr_decay = 0.95
    conv_non_linear = "relu"
    sqr_norm_lim = 9
    shuffle_batch = True
    
    print(f"\nTraining config:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Sentence length: {img_h}")
    print(f"  Classes: 4")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    
    # 数据打乱
    if shuffle_batch:
        np.random.shuffle(train_data)
    
    # 准备 batch
    if train_data.shape[0] % batch_size > 0:
        extra = batch_size - train_data.shape[0] % batch_size
        train_data = np.vstack([train_data, train_data[:extra]])
    
    n_batches = train_data.shape[0] // batch_size
    n_train_batches = int(n_batches * 0.9)
    
    train_set = train_data[:n_train_batches*batch_size]
    val_set = train_data[n_train_batches*batch_size:]
    
    train_set_x, train_set_y = train_set[:, :img_h], train_set[:, -1]
    val_set_x, val_set_y = val_set[:, :img_h], val_set[:, -1]
    test_set_x, test_set_y = test_data[:, :img_h], test_data[:, -1]
    
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    val_set_x, val_set_y = shared_dataset(val_set_x, val_set_y)
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    
    n_val_batches = val_set.shape[0] // batch_size
    
    # 构建模型
    print("\nBuilding model...")
    rng = np.random.RandomState(3435)
    
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = [(feature_maps, 1, fh, filter_w) for fh in filter_hs]
    pool_sizes = [(img_h - fh + 1, img_w - filter_w + 1) for fh in filter_hs]
    
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()
    
    Words = theano.shared(value=U, name="Words")
    layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((batch_size, 1, img_h, img_w))
    
    conv_layers = []
    layer1_inputs = []
    
    for i, (filter_shape, pool_size) in enumerate(zip(filter_shapes, pool_sizes)):
        conv_layer = LeNetConvPoolLayer(
            rng, input=layer0_input,
            image_shape=(batch_size, 1, img_h, img_w),
            filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear
        )
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    
    layer1_input = T.concatenate(layer1_inputs, 1)
    
    classifier = MLPDropout(
        rng, input=layer1_input,
        layer_sizes=[feature_maps*len(filter_hs), hidden_units[1]],
        activations=[Iden], dropout_rates=dropout_rate
    )
    
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        params += [Words]
    
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    # 编译函数
    print("Compiling functions...")
    
    train_model = theano.function(
        [index], cost, updates=grad_updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )
    
    val_model = theano.function(
        [index], classifier.errors(y),
        givens={
            x: val_set_x[index*batch_size:(index+1)*batch_size],
            y: val_set_y[index*batch_size:(index+1)*batch_size]
        }
    )
    
    test_model = theano.function(
        [index], classifier.errors(y),
        givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )
    
    # 训练
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_perf = 0
    best_params = None
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # 训练
        if shuffle_batch:
            for idx in np.random.permutation(range(n_train_batches)):
                train_model(idx)
        else:
            for idx in range(n_train_batches):
                train_model(idx)
        
        # 验证
        val_losses = [val_model(i) for i in range(n_val_batches)]
        val_perf = 1 - np.mean(val_losses)
        
        # 测试
        test_losses = [test_model(i) for i in range(test_data.shape[0]//batch_size)]
        test_perf = 1 - np.mean(test_losses)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{n_epochs}: val_acc={val_perf:.4f}, test_acc={test_perf:.4f}, time={epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            best_params = [p.get_value() for p in classifier.params]
    
    # 保存模型
    print(f"\nSaving model to {OUTPUT_MODEL}...")
    os.makedirs('save', exist_ok=True)
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(best_params, f)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_perf:.4f}")
    print(f"Model saved to: {OUTPUT_MODEL}")
    print(f"Data saved to: {OUTPUT_PICKLE}")
    print("="*60)

