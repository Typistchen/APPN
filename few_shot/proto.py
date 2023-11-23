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
from few_shot.datasets import TransformLoader
def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      k_shot: int,
                      n_way: int,
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
    
    
    if (noise_type != None):
        x, noise_index = add_noise(k_shot, n_way, x, noise_type , class_name, total_class, dataset, noise_dataset)

    confidence_y = torch.zeros(n_way*(k_shot + q_queries))
    confidence_y[noise_index] = 1
    confidence_y = confidence_y.long().cuda()
    normalize_confidence, label_all_clean = model(x) # 45 10 10 3
    
    pre_query_label = label_all_clean[ -k_shot * q_queries : ,:n_way]

    loss1 = loss_fn(normalize_confidence, confidence_y)
    loss2 = loss_fn(pre_query_label, y)
    loss = loss1 + loss2 

    # Prediction probabilities are softmax over distances
    y_pred = pre_query_label

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
    
    return loss2, y_pred

def add_noise(k_shot, n_way, data, noise_type, class_name, total_class, dataset, noisedataset, model):
    transform_init = TransformLoader(84)
    if model == "val":
        transform = transform_init.get_composed_transform(aug=False)
    elif model == "train":
        transform = transform_init.get_composed_transform(aug=True)


    if noise_type == 'IT':
        noise_idxes = []
        for i in range(n_way):
            noise_idxes.append(k_shot * i  + random.randint(0, 4))
        temp = data[noise_idxes[0]]
        for i in range(len(noise_idxes)-1):
            data[noise_idxes[i]] = data[noise_idxes[i+1]]
        data[noise_idxes[-1]] = temp
        return data, noise_idxes
    
    elif noise_type == 'OOT':
        class_name = list(set(class_name)) # 去掉重复元素
        className_key = list(total_class.keys())
        for name in class_name:
            className_key.remove(name) # 删除指定元素
        selected_images = []
        random_images = random.sample(className_key, n_way) #随机选取5个键值
        for key_name in random_images:
            picture_name = random.sample(total_class[key_name], 1)
            for picname in picture_name:
                path = DATA_PATH + '/' + dataset + '/train/' + key_name + '/' + picname
                # "/home/dsz/cjq/few-shot-master/data/fc100/train/train/train_s_002445.png"
                
                image = Image.open(path)
                transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                tensor_image = transform(image)
                selected_images.append(tensor_image)

        noise_idxes = []
        for i in range(n_way):
            noise_idxes.append(k_shot * i  + random.randint(0, 4))
        for i in range(len(noise_idxes)):
            data[noise_idxes[i]] = selected_images[i]
        return data,noise_idxes

    elif noise_type == 'OOD':
        className_key = list(total_class.keys())
        random_images = random.sample(className_key, n_way) #随机选取5个键值
        selected_images = []
        for key_name in random_images:
            picture_name = random.sample(total_class[key_name], 1)
            for picname in picture_name:
                if model == "train":
                    path = DATA_PATH + '/' + noisedataset + '/train/' + key_name + '/' + picname
                if model == "val":
                    path = DATA_PATH + '/' + noisedataset + '/val/' + key_name + '/' + picname
                image = Image.open(path)
                # if dataset == 'miniImageNet' or dataset == 'tieredImagenet':
                #     transform = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Resize(84)
                #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #     #             std=[0.229, 0.224, 0.225])
                #     ])
                #     tensor_image = transform(image)
                #     selected_images.append(tensor_image)
                # elif dataset == 'cifar' or dataset == 'fc100':
                #     transform = transforms.Compose([
                #     transforms.CenterCrop(350),
                #     transforms.ToTensor(),
                #     transforms.Resize(84)
                #     # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #     #                 std=[0.5, 0.5, 0.5])
                #     ])
                #     tensor_image = transform(image)
                #     selected_images.append(tensor_image)

                tensor_image = transform(image)
                selected_images.append(tensor_image)



        noise_idxes = []
        for i in range(n_way):
            noise_idxes.append(k_shot * i  + random.randint(0, 4))
        for i in range(len(noise_idxes)):
            data[noise_idxes[i]] = selected_images[i]
        return data,noise_idxes
    elif noise_type == False:
        return data