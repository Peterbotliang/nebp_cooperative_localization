# NEBP Cooperative Localization

This repository is the python implementation of the following paper

-  M. Liang and F. Meyer, “Neural enhanced belief propagation for
cooperative localization,” in Proc. IEEE SSP-21, Jul. 2021, pp. 326-330. [[Arxiv](https://arxiv.org/abs/2105.12903)] [[IEEE](https://ieeexplore.ieee.org/document/9513853)]

## Requirements

- pytorch==1.7.1, 2.1.2
- dgl==0.5.3, 2.0.x
- numpy>=1.18.1

The package versions listed above are the ones that has been tested on. Other versions may also work.

The code has been tested on ubuntu 18.04, cuda 10.2 and ubuntu 22.04, cuda 12.2.

## Usage

To train the NEBP cooperative localization neural network:

```
python train.py
```

Check the parameters that can be tuned using `python train.py --help`. It takes some time to generate the synthetic data.

For testing (including both BP and NEBP), use

```
python test.py
```

Check the parameters that can be tuned using `python test.py --help`
