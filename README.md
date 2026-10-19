# Fine-Grained Preference Data Selection

An instruction for generating and selecting data for DPO.

## Requirements

*   Python 3.10
*   PyTorch 2.2.0
*   AlpacaEval (https://github.com/tatsu-lab/alpaca_eval)
*   Framework based on LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory)

## Usage

The process consists of two main steps:

**1. Generate Aggregated Data with varying preferece conflicts**

This script prepares the aspect-based fine-grained preference dataset.

```bash
python main/prepare/dpodata_gen_aspects.py
```

**2. Run Selection and Training Pipeline**

This script executes the main workflow, including data selection and model training.

```bash
python main/pipeline.py
```