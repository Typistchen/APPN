import sys
root_path = "/home/dsz/Documents/cjq/few-shot-master"
sys.path.append(root_path)

from torch.utils.data import DataLoader
import argparse
import model
from few_shot.datasets import TrainDataset
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode,add_noise
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from model import CNNEncoder
from config import PATH, DATA_PATH
from utils.utils import *
import json
import torch
import torchvision.models as models
import torch.nn as nn


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True



matrix_size = 105
symmetric_matrix = torch.zeros((matrix_size, matrix_size))
symmetric_matrix.fill_diagonal_(1)
for m in range(5):
    for i in range(5):
        for j in range(5):
            symmetric_matrix[i+m*5, j+m*5] = symmetric_matrix[j+m*5, i+m*5] = 1  # 这里用随机数来填充非对角线元素
        for x in range(16):
            symmetric_matrix[i+m*5, 25 + x +m*16] = symmetric_matrix[x + 25+m*16, i+m*5] = 1  # 这里用随机数来填充非对角线元素
for t in range(5):
    for p in range(16):
        for q in range(16):
            symmetric_matrix[p+25+t*16, 25+q+t*16] = symmetric_matrix[ p+25+t*16,25+q+t*15] = 1  # 这里用随机数来填充非对角线元素

edge = symmetric_matrix.type(torch.float32).cuda() 
flat_array = edge.ravel().cuda()
flat_array = flat_array.long()
print(edge)