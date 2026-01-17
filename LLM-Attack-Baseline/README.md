# LLM-Attack-Baseline

## Hotflip
```bash
cd Hotflip
python attack_c4_llm.py
```

## UAT
```bash
cd UAT/c4
python c4_attack.py --model qwen
python c4_attack.py --model llama
python c4_attack.py --model mistral
python c4_attack.py --model deepseek
```

## AutoPrompt
```bash
cd AutoPrompt
python c4_attack.py --model qwen --num_samples 1000
python c4_attack.py --model llama --num_samples 1000
python c4_attack.py --model mistral --num_samples 1000
python c4_attack.py --model deepseek --num_samples 1000
```

## GCG
```bash
cd GCG
python c4_attack.py --model qwen --num_samples 1000
python c4_attack.py --model llama --num_samples 1000
python c4_attack.py --model mistral --num_samples 1000
python c4_attack.py --model deepseek --num_samples 1000
```