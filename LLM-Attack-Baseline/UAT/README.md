sst2:
cd ~/universal-triggers-master/sst
python sst.py

agnews:
cd ~/universal-triggers-master/agnews
python agnews.py

c4:
cd ~/universal-triggers-master/c4
# Qwen
python c4_attack.py --model qwen 
# Llama
python c4_attack.py --model llama
# Mistral
python c4_attack.py --model mistral
# Deepseek
python c4_attack.py --model deepseek

eli5:
cd ~/universal-triggers-master/eli5
# Qwen
python eli5_attack.py --model qwen 
# Llama
python eli5_attack.py --model llama 
# Mistral
python eli5_attack.py --model mistral 
# Deepseek
python eli5_attack.py --model deepseek 
