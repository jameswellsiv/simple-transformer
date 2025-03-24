# Simple Transformer

This project is a simple implementation of a Transformer model. The main goal of this project is to help people understand the mathematics behind the Attention mechanism in Transformer models.

## Overview

Transformers have revolutionized the field of natural language processing by introducing the concept of self-attention, which allows the model to weigh the importance of different words in a sentence. This project provides a straightforward implementation of a Transformer model, focusing on the core concepts and mathematical foundations of Attention.

## Project Structure

- `data/`: Contains the dataset used for training and inference.
  - `shakespeare.txt`: Example dataset.
- `src/`: Source code for the project.
  - `simple_transformer/`: Contains the implementation of the Transformer model.
    - `__init__.py`: Initialization file for the module.
    - `transformer/`: Contains the pytorch modules for a Transformer.
        - `attention.py`: Pytorch module for Attention mechanism
        - `positional_encoding.py`: Code for creating positional encodings
        - `transformer_block`: Pytorch Module for the whole transformer block (i.e. Attention and FFN)
        - `model.py`: Pytorch model defintion for Transformer (decoder only)
    - `inference.py`: Script for running inference with the trained model.
    - `train.py`: Script for training the Transformer model.

## Getting Started

### Installing Dependencies

This project uses `uv` to manage dependencies. To install the required dependencies, run the following command:

```sh
uv install
```

### Running the Project

To get started with this project, you can explore the `train.py` and `inference.py` scripts in the `src/simple_transformer/` directory. These scripts provide a basic implementation of training and inference using the Transformer model.

Also, each python file has a self contained example that you can run to further your understanding of a specific step in the process.