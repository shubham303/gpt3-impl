# GPT-Character-Level-Implementation

This is a character-level implementation of the GPT (Generative Pre-training Transformer) model. The program consists of two main files: `model.py` and `train.py`.

`model.py` contains the code for the GPT model, including the architecture and the forward pass.

`train.py` contains the boilerplate code for training the model on a given dataset. The training script accepts a number of command-line arguments, including the batch size, block size, number of training iterations, evaluation interval, learning rate, device, and number of evaluation iterations. The command to run the program is:
```sh
python train.py --use_wandb True --batch_size 192 --block_size 256 --max_iters 5000 --eval_interval 500 --learning_rate 0.0009 --device cuda --eval_iters 200 --n_embd 384 --n_head 2 --n_layer 3 --dropout 0.2

```
The input.txt is the text dataset file that will be used to train the model. The dataset should be preprocessed and cleaned before using it.

The train.py script uses wandb library to log the training progress and hyperparameters, if you want to disable wandb just set use_wandb argument to False

It's recommended to run the script on GPU for faster training, but if it's not available then you can use CPU by setting the device argument to 'cpu'

You can fine tune the hyperparameters to get better results, it's always good to start with small values and then increase them gradually.

The code is well commented and easy to understand, you can also modify it to suit your needs.