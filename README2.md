# APPN: a Attention-based Pseudo-label Propagation Network for few-shot learning with noisy labels

The aim for this repository is to contain clean, readable and tested
code to produce APPN.

This project is written in python 3.6 and Pytorch and assumes you have
a GPU.

# Setup


# 

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

