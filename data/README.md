# Data

Dataset files and loading utilities.

## Files

- `processed_c4.json` - C4 dataset for target generation
- `sst2_clean.tsv` - SST-2 sentiment classification
- `agnews_clean.tsv` - AG News topic classification  
- `datasets.py` - Dataset loading utilities
- `data.py` - Dataset preprocessing script

## Usage

### Generate TSV files

```bash
cd data
python data.py
```

Downloads SST-2 and AG News from HuggingFace and generates TSV files (1000 samples each).

### Load data

```python
from data.datasets import load_texts, load_full_data

# Load text only
texts = load_texts(["c4"], samples_per_ds=100)

# Load full samples
data = load_full_data(["sst2", "agnews"], samples_per_ds=1000)
```

## Data Formats

**C4 (JSON Lines)**:
```json
{"prompt": "...", "natural_text": "..."}
```

**TSV**:
```
label\ttext
0\tThis movie is terrible
```
