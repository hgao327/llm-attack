# Adversarial Attack Experiments

Implementation of **AutoPrompt** and **GCG** attacks for classification and target word generation tasks.

## Structure

```
icml experiment/
├── classification/              # Classification attacks
│   ├── autoprompt_classification.py
│   └── gcg_classification.py
│
├── target_generation/          # Target word generation attacks
│   ├── autoprompt_generate.py
│   └── gcg_generate.py
│
├── data/                       # Datasets
│   ├── processed_c4.json
│   ├── sst2_clean.tsv
│   ├── agnews_clean.tsv
│   └── datasets.py
│
├── run_experiments.py         # Main runner
└── outputs/                   # Results
```

## Tasks

### Classification
- **Goal**: Flip model predictions
- **Datasets**: SST-2 (sentiment), AG News (topic)
- **Methods**: AutoPrompt (BERT MLM), GCG (fine-tuned classifiers)

### Target Word Generation
- **Goal**: Force model to generate "idiot"
- **Dataset**: C4 (100 samples)
- **Methods**: AutoPrompt (BERT MLM), GCG (GPT-2)

## Quick Start

```bash
# Target word generation
python target_generation/autoprompt_generate.py
python target_generation/gcg_generate.py

# Or use the runner
python run_experiments.py --task target_generation
python run_experiments.py --all
```

## Output Files

```
outputs/
├── autoprompt_target_idiot.txt
├── gcg_target_idiot.txt
├── autoprompt_sst2.txt
├── autoprompt_agnews.txt
├── gcg_sst2.txt
└── gcg_agnews.txt
```

## Requirements

```bash
pip install torch transformers datasets tqdm numpy
```

## References

- AutoPrompt: Shin et al. (2020)
- GCG: Zou et al. (2023)
