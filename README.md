# Relation Classification via Convolutoinal Deep Neural Network

## Introduction
A re-implementation of the paper(http://www.aclweb.org/anthology/C14-1220)
## Dataset
SemEval-2010 task8
## How to run ?
1. For train: $ python3 run.py --train
2. For test: $python3 run.py --test
3. For evaluation: $ bash eval.sh
## Requirements
1. python3
2. tensorflow >= 1.5
3. download GloVe (https://github.com/stanfordnlp/GloVe)
## Result
F-score is about 80% ± 0.5%
## Reference
https://github.com/FrankWork/conv_relation
