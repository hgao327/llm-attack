sst2:  python use_pretrained_gene_testset.py -nonstatic -word2vec
agnews:
# 步骤1: 训练 AG News 模型（如果没有预训练模型）
python train_agnews_full.py -nonstatic -word2vec
# 步骤2: 生成对抗样本
python generate_agnews_adv.py -nonstatic -word2vec

c4:
# 运行（测试所有模型）
python attack_c4_llm.py
# 或者只测试单个模型
python attack_c4_llm.py --model Qwen/Qwen1.5-1.8B-Chat

eli5:
# 运行（测试所有模型）
python attack_eli5_refusal.py
# 或者只测试单个模型
python attack_eli5_refusal.py --model Qwen/Qwen1.5-1.8B-Chat
