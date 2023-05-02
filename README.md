# Class-Incremental Learning with Causal Relational Replay (CRR)

This is the Pytorch implementation of the paper: Class-Incremental Learning with Causal Relational Replay.

## Installation and Usage.
1. Install the dependencies
```bash
pip install -r requirements.txt
```
2. Use ./utils/main.py to run experiments.

```bash
python ./utils/main.py --model crr --seed 2711 --end_lr 0.4 --dataset seq-cifar10 --buffer_size 600 --lr 0.9 --batch_size 64 --minibatch_size 64 --batch_size_test 128 --n_epochs 50 --alpha 0.75 --beta 1.75 --gamma 1.25 --csv_log
```

```bash
python ./utils/main.py --model crr --seed 2711 --end_lr 0.32 --dataset seq-cifar100 --buffer_size 600 --lr 0.9 --batch_size 64 --minibatch_size 64 --batch_size_test 128 --n_epochs 50 --alpha 0.75 --beta 1.75 --gamma 1.25 --csv_log
```

```bash
python ./utils/main.py --model crr --seed 2711 --end_lr 0.36 --dataset seq-core50 --buffer_size 600 --lr 0.9 --batch_size 48 --minibatch_size 48 --batch_size_test 48 --n_epochs 15 --alpha 0.75 --beta 1.75 --gamma 1.25 --csv_log
```
## Acknowledge: 

Mammoth - An Extendible (General) Continual Learning Framework for Pytorch ([link](https://github.com/aimagelab/mammoth)).
