# FR

Code for paper "Frequency Regularization for Improving Adversarial Robustness" accepted by Practical-DL Workshop @AAAI 2023.

## Train

```bash

# Standard adversarial training
# dataset_path: Path for the dataset.
# trial: Experiment index
python main.py --model resnet --dataset cifar10 --dataset_path XXXXX --trial X

# AT + FR
python main.py --model resnet --dataset cifar10 --dataset_path XXXXX --trial X --fre_loss

# AT + FR/WA
python main.py --model resnet --dataset cifar10 --dataset_path XXXXX --trial X --fre_loss --swa

```

## Evaluation
```bash
# ckpt_path: Path for the saved checkpoint.
python Robustness.py --ckpt_path XXXXXXXXX
```
