# -*- coding: utf-8 -*-
"""
数据预处理脚本：将 SST-2 数据转换为所需格式
"""
import os
import sys

def convert_sst2_data():
    """
    将 stsa.binary 格式转换为 process_data_sst2.py 需要的格式
    格式: 每两行一组，第一行是句子，第二行是标签（positive/negative）
    """
    train_file = 'SentDataPre-master/data/sst2/stsa.binary.train'
    test_file = 'SentDataPre-master/data/sst2/stsa.binary.test'
    
    train_out = 'SentDataPre-master/data/sst2/sst2_train.txt'
    test_out = 'SentDataPre-master/data/sst2/sst2_test.txt'
    
    print("转换训练集...")
    with open(train_file, 'r') as f_in, open(train_out, 'w') as f_out:
        for line in f_in:
            label = line[0]  # 0 或 1
            sentence = line[2:].strip()  # 去掉标签和空格
            label_text = 'positive' if label == '1' else 'negative'
            f_out.write(sentence + '\n')
            f_out.write(label_text + '\n')
    
    print("转换测试集...")
    with open(test_file, 'r') as f_in, open(test_out, 'w') as f_out:
        for line in f_in:
            label = line[0]
            sentence = line[2:].strip()
            label_text = 'positive' if label == '1' else 'negative'
            f_out.write(sentence + '\n')
            f_out.write(label_text + '\n')
    
    print("数据转换完成！")
    print(f"训练集: {train_out}")
    print(f"测试集: {test_out}")

def generate_pickle():
    """
    生成 sst2.p 文件
    """
    # 保存当前目录
    original_dir = os.getcwd()
    
    # 切换到 SentDataPre-master/cnn 目录
    cnn_dir = os.path.join(original_dir, 'SentDataPre-master', 'cnn')
    os.chdir(cnn_dir)
    
    # 添加到路径
    sys.path.insert(0, cnn_dir)
    
    try:
        from process_data_sst2 import process_data
        
        # 输出文件在项目根目录
        output_file = os.path.join(original_dir, 'sst2.p')
        print(f"\n生成 {output_file}...")
        process_data(output_file)
        print(f"{output_file} 生成完成！")
    finally:
        # 切换回原目录
        os.chdir(original_dir)

if __name__ == '__main__':
    # 步骤1: 转换数据格式
    convert_sst2_data()
    
    # 步骤2: 生成 pickle 文件
    generate_pickle()
    
    print("\n✅ 数据准备完成！现在可以运行 use_pretrained_gene_testset.py 了")

