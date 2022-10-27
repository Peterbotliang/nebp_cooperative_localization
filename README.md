# Cooperative Localization

This repository is the python implementation of the following paper

-  M. Liang and F. Meyer, “Neural enhanced belief propagation for
cooperative localization,” in Proc. IEEE SSP-21, Jul. 2021, pp. 326-330. [[Arxiv](https://arxiv.org/abs/2105.12903)] [[IEEE](https://ieeexplore.ieee.org/document/9513853)]

## Requirements

- pytorch==1.7.1
- dgl==0.5.3
- numpy>=1.18.1

The code is tested on ubuntu 18.04, cuda 10.2.

## Usage

Run the cooperative localization algorithm:

```
python main.py
```

Check the parameters that can be tuned using `python main.py --help`