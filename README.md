# ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall

[![arXiv](https://img.shields.io/badge/arXiv-2510.07896-b31b1b.svg)](https://arxiv.org/abs/2510.07896)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)

This is the official implementation of the paper "ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall" (ICLR 2026).

## Overview

Large language models (LLMs) face challenges in multi-hop factual reasoning when knowledge editing is required. Through causal analysis, we discover that existing methods overlook the neuron-level dynamic representations of implicit subjects in reasoning chains. We propose **ACE** (Attribution-Controlled Editing), a framework that leverages neuron-level attribution to identify and edit critical query-value neuron pathways.

**Key Results:**
- **+9.44%** improvement on GPT-J
- **+37.46%** improvement on Qwen3-8B

## Project Structure

```
ACE/
├── ace/
│   ├── edit/                    # Knowledge editing implementation
│   │   ├── ace/                 # Core ACE algorithm
│   │   │   ├── ace_main.py      # Main editing logic
│   │   │   ├── ace_hyper.py     # Hyperparameters
│   │   │   ├── compute_ks.py    # Key computation
│   │   │   └── compute_zs.py    # Value computation
│   │   ├── hparams/             # Model-specific hyperparameters
│   │   ├── evaluate_filterqa_layer.py  # Evaluation script
│   │   └── summarize.py         # Results summarization
│   ├── knowledge-prob/          # Knowledge probing utilities
│   ├── main_gptj_r.ipynb        # GPT-J experiments
│   ├── main_qwen3_r.ipynb       # Qwen3-8B experiments
│   └── run_layer.sh             # Running script
└── sample_data.json             # Sample data for testing
```

## Installation

```bash
git clone https://github.com/yjywdzh/ACE.git
cd ACE
pip install -r requirements.txt
```

## Usage

### Running ACE on Qwen3-8B

```bash
cd ace
bash run_layer.sh
```

Or run with custom parameters:

```bash
python ./edit/evaluate_filterqa_layer.py \
  --model_path path/to/model \
  --model_name Qwen/Qwen3-8B \
  --alg_name ace \
  --hparams_fname qwen3_8b.json \
  --ds_name filteredQA \
  --num_edits 1 \
  --use_cache
```
*You can use your custom model to run the experiments, after capturing the neurons/layers using jupyter notebooks to do the interpretability part.*

### Jupyter Notebooks

For interactive experiments:
- `ace/main_gptj_r.ipynb` - GPT-J experiments
- `ace/main_qwen3_r.ipynb` - Qwen3-8B experiments

## Citation

```bibtex
@inproceedings{yang2026ace,
  title={ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall},
  author={Yang, Jiayu and Fan, Yuxuan and Lai, Songning and Wu, Shengen and Tang, Jiaqi and Kang, Chun and Guo, Zhijiang and Yue, Yutao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [PMET](https://github.com/xpq-tech/PMET) for their inspiring work.

## License

This project is released under the MIT License.
