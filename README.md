## RoBerta Fine Tuning with LoRA
This repository is the the submission for the NYU CS 6953 Project 2

## Introduction

This project explores parameter-efficient fine-tuning of the **RoBERTa** transformer model on the **AG News** classification dataset using **Low-Rank Adaptation (LoRA)**

## Repository
- main.py: File that downloads the dataset, defines the LoRA config and trains the model
- Python Notebooks: Folder containing .ipynb files

---
## Prerequisites

Ensure you have **Python 3.6+** installed before running the project.

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

---

## Training the Model

Run the following command to start training:

```sh
python main.py
```

This will:

- Download and preprocess the AG-News Dataset.
- Train the model with the specified hyperparameters.
- Save the model checkpoints in `./results/`.

---

## Hyperparameters

Below is the summary of the key hyperparameters used in training:

## 🔧 Hyperparameters

| **Hyperparameter**         | **Value**                      |
|---------------------------|--------------------------------|
| Optimizer                 | ADAM                           |
| Learning Rate (LR)        | 0.0005                         |
| LR Scheduler              | CosineAnnealing                |
| Weight Decay              | 0.01                           |
| Epochs                    | 4                              |
| Batch Size                | 128                            |
| LoRA Droupout             | 0.1                            |
| fp16                      | True                           |


## Model Checkpoints

```
./results/
```

## Results

The model achieves a **validation accuracy of 94%**, and **test accuracy on unseen data of 85.37%**

---

## References

- Kamp, Mariano. “A Winding Road to Parameter Efficiency.” A Winding Road to Parameter Efficiency, https://towardsdatascience.com/, https://towardsdatascience.com/a-winding-road-to-parameter-efficiency-12448e64524d/. Accessed 20 April 2025.
- Raschka, Sebastian, and Cameron R. Wolfe. “Easily Train a Specialized LLM: PEFT, LoRA, QLoRA, LLaMA-Adapter, and More.” Deep (Learning) Focus, 27 November 2023, https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft. Accessed 20 April 2025.
- Valizadeh Aslani, Taha. “LayerNorm: A key component in parameter-efficient fine-tuning.” arXiv, https://arxiv.org/html/2403.20284v1. Accessed 20 April 2025.
- Hu, Edward, et al. “[2106.09685] LoRA: Low-Rank Adaptation of Large Language Models.” arXiv, 17 June 2021, https://arxiv.org/abs/2106.09685. Accessed 20 April 2025.