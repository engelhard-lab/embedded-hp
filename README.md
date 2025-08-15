# Balancing Interpretability and Flexibility in Modeling Diagnostic Trajectories with an Embedded Neural Hawkes Process Model

## Introduction

This repository contains the source code for the paper "Balancing Interpretability and Flexibility in Modeling Diagnostic Trajectories with an Embedded Neural Hawkes Process Model". We propose a novel deep learning model for evaluating the performance of Hawkes processes on event sequence data.

Hawkes processes are a type of self-exciting point process where the occurrence of an event increases the likelihood of future events. They are widely used for modeling event data in various domains, including social media, finance, seismology, and more. This project provides an implementation of our proposed model and the necessary tools to reproduce the experiments in our paper.


## Dataset

The dataset used for this project can be found at the following link:

[Dataset](https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7)

The dataset consists of event sequences, where each event has a timestamp and a type (or mark). The dataset is from [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess). 

## Usage

### Training

To train the model, run the `train.py` script. You can specify various hyperparameters as command-line arguments.

```bash
python train.py --data_path [path_to_your_dataset] --epochs [number_of_epochs] --learning_rate [your_learning_rate]
```

For a full list of command-line arguments, please see the `train.py` script or run:
```bash
python train.py --help
```

### Sampler

The sampling predecdure is from [Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes) for next event prediction. 


## Code Structure

* `data_loader.py`: Contains the data loading and preprocessing logic.
* `model.py`: Defines the architecture of our proposed neural network model.
* `train.py`: The main script for training the model.
* `sampler.py`: A script for sampling the next event. 

## Citation

If you find this code useful in your research, please consider citing our paper:

```
[Zhao, Y & Engelhard, M. M. (2025). Balancing Interpretability and Flexibility in Modeling Diagnostic Trajectories with an Embedded Neural Hawkes Process Model. Proceedings of Machine Learning Research, 298, 1-26.]
```

## Contact

For any questions or suggestions, please feel free to open an issue. 
