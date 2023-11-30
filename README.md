# APPN: a Attention-based Pseudo-label Propagation Network for few-shot learning with noisy labels

The aim for this repository is to contain clean, readable and tested
code to reproduce few-shot learning research.

This project is written in python 3.6 and Pytorch and assumes you have
a GPU.

See these Medium articles for some more information
1. [Theory and concepts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
2. [Discussion of implementation details](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d)

# Setup
### Requirements

Listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

### Data
Edit the `DATA_PATH` variable in `config.py` to the location where
you store the Omniglot and miniImagenet datasets.

After acquiring the
data and running the setup scripts your folder structure should look
like
```
DATA_PATH/
    Omniglot/
        images_background/
        images_evaluation/
    miniImageNet/
        images_background/
        images_evaluation/
```

**Omniglot** dataset. Download from https://github.com/brendenlake/omniglot/tree/master/python,
place the extracted files into `DATA_PATH/Omniglot_Raw` and run
`scripts/prepare_omniglot.py`

**miniImageNet** dataset. Download files from
https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view,
place in `data/miniImageNet/images` and run `scripts/prepare_mini_imagenet.py`

### Tests (optional)

After adding the datasets run `pytest` in the root directory to run
all tests.

# Results

The file `experiments/experiments.txt` contains the hyperparameters I
used to obtain the results given below.


Number in brackets indicates 1st or 2nd order MAML.# few_shot
# APPN
