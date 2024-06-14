import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable
from few_shot.utils import pairwise_distances
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from torchvision import transforms
from PIL import Image
from config import DATA_PATH

def proto_net_episode_gcn(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool,
                      class_name = None,
                      noise_type = False,
                      total_class = None,
                      dataset = None,
                      noise_dataset = None,
                      ):
    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update
        class_name: The categories already included in the training process
        noise_type: The type of added noise

    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    x = x
    y = y 
    class_name = class_name
    if (train) and (noise_type != None):
        x = add_noise(n_shot, k_way, x, noise_type , class_name, total_class, dataset, noise_dataset)
    embeddings = model(x) # 45 10 10 3

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]
    prototypes = compute_prototypes_gcn(support, k_way, n_shot)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass
    # image_tensor = x.cpu()
    # image_tensor = np.array(image_tensor)
    # #image_tensor = np.multiply(image_tensor, 255)
    # data_reshaped = image_tensor.reshape((360, 32, 32, 3))

    # for i in range(360):    
    #     plt.imshow(data_reshaped[i], interpolation='nearest')
    #     path = "test_p" + "/output" + str(i) + '.png'
    #     plt.savefig(path) # 保存图像到output.png文件中
    
    return loss, y_pred


def compute_prototypes_gcn(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes

def add_noise(n_shot, k_way, data, noise_type, class_name, total_class, dataset, noisedataset):
    if noise_type == 'IT':
        noise_idxes = []
        for i in range(k_way):
            noise_idxes.append(n_shot * i  + random.randint(0, 4))
        temp = data[noise_idxes[0]]
        for i in range(len(noise_idxes)-1):
            data[noise_idxes[i]] = data[noise_idxes[i+1]]
        data[noise_idxes[-1]] = temp
        return data
    
    elif noise_type == 'OOT':
        class_name = list(set(class_name)) # 去掉重复元素
        className_key = list(total_class.keys())
        for name in class_name:
            className_key.remove(name) # 删除指定元素
        selected_images = []
        random_images = random.sample(className_key, k_way) #随机选取5个键值
        for key_name in random_images:
            picture_name = random.sample(total_class[key_name], 2)
            for picname in picture_name:
                path = DATA_PATH + '/' + dataset + '/train/' + key_name + '/' + picname
                image = Image.open(path)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                tensor_image = transform(image)
                selected_images.append(tensor_image)

        noise_idxes = []
        for i in range(k_way):
            noise_idxes.append(n_shot * i  + random.randint(0, 4))
        for i in range(len(noise_idxes)):
            data[noise_idxes[i]] = selected_images[i]
        return data

    elif noise_type == 'OOD':
        className_key = list(total_class.keys())
        random_images = random.sample(className_key, k_way) #随机选取5个键值
        selected_images = []
        for key_name in random_images:
            picture_name = random.sample(total_class[key_name], 1)
            for picname in picture_name:
                path = DATA_PATH + '/' + noisedataset + '/train/' + key_name + '/' + picname
                image = Image.open(path)
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                tensor_image = transform(image)
                selected_images.append(tensor_image)

        noise_idxes = []
        for i in range(k_way):
            noise_idxes.append(n_shot * i  + random.randint(0, 4))
        for i in range(len(noise_idxes)):
            data[noise_idxes[i]] = selected_images[i]
        return data
    elif noise_type == False:
        return data