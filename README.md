# FEDORA
An implementation for "[Personalized Federated Learning with Parameter Propagation](https://dl.acm.org/doi/abs/10.1145/3580305.3599464?casa_token=iNEG1gXvM9MAAAAA:hEVl21M1Vlh3ZJVTg-iaJZnUcx3RsVX_OmZ4oVPQ3nSZ1TVyQMWtnYT7glRgaIGGQY3bG93-lpnL)" (KDD'23).

## Environment Requirements
The code has been tested under Python 3.7.4. The required packages are as follows:
* numpy==1.21.6
* torch==1.13.1+cu117
* torchvision==0.14.1+cu117
* tqdm==4.66.1

## Acknowledgement
This is the latest source code of **FEDORA** for KDD-2023. If you find that it is helpful for your research, please consider citing our paper:

```
@inproceedings{wu2023personalized,
  title={Personalized Federated Learning with Parameter Propagation},
  author={Wu, Jun and Bao, Wenxuan and Ainsworth, Elizabeth and He, Jingrui},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2594--2605},
  year={2023}
}
```

## Reference
Some codes of **FEDORA** are adapted from the following baselines.

LG-FedAvg: https://github.com/pliang279/LG-FedAvg

pFedHN: https://github.com/AvivSham/pFedHN
