# tdl-express-distributions

## Introduction

This repository contains the code for the experiments and improvements based on the paper: On the ability of neural nets to express distributions ([here](https://arxiv.org/abs/1702.07028)).

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Experiments

The experiments on the report were run from both notebooks available.
You need to define a config for the experiment you want to run:

```python
config = {
    "function_distrib":{
        "n": 10000, # number of data generated for the visualizations and the validation
        "d": 2, # input dimension
        "l": 8, # number of functions composed (only useful if random is set to True)
        "output_size": 1, # output dimension
        "function_type": "linear-sigmoid-trigonometric", # functions to compose
        "random": True # when set to None, will exactly compose the functions above in the same order, otherwise randomly sample l functions from the above list
        "output_function": "linear", # choose which final function is used
        "prior": "gaussian_mixture", # uniform_cube, uniform_ball, gaussian, gaussian_mixture
        "seed": 1,
    },

    "model": {
        # "input_size": # infer
        # "output_size": # infer
        "hidden_size": 100, # number of neurons in every hidden layer of the neural network
        "n_hidden_layers": 4 # number of layers of the neural network
    },

    "optimizer": {
        "lr": 0.01, # learning rate
        "batch_size": 32, # batch sized sampled at every epoch from the prior distribution
    },

    "epochs": 3000,
}

```

In order to run an experiment and plot the results, you then need to define the trainer and train it:

```python
from src.trainer import Trainer
trainer = Trainer(config)
losses, w2_distances = trainer.train()
trainer.plot_input_output(title="Test experiment")
```
