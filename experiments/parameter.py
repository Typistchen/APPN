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
import torchsummary as summary

from few_shot.models import FewShotClassifier
setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda:1')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############


parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't')  # Quick hack to extract boolean
parser.add_argument('--distance', default='cosine')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--lstm-layers', default=1, type=int)
parser.add_argument('--unrolling-steps', default=2, type=int)
args = parser.parse_args()

###################
# Create datasets #
###################


args = parser.parse_args()
print(args)


# dataset_class = TrainDataset
# train = dataset_class('train', args.dataset) 
# train_taskloader = DataLoader(
#     train,
#     batch_sampler=NShotTaskSampler(train, episodes_per_epoch, args.k_shot, args.n_way, args.n_query),
#     num_workers=4
# )
# evaluation = dataset_class('val', args.dataset)
# evaluation_taskloader = DataLoader(
#     evaluation,
#     batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.k_shot, args.n_way, args.n_query),
#     num_workers=4
# )

#########
# Model # 
#########



###########
# Training #
###########
 
# save model
best_val_accuracy = 0.0
best_model = None

# 进行训练
num_epochs = 300

lr=1e-5
# backbone = CNNEncoder(args)
# model1 = model.LabelPropagation(args, backbone)


model2 = meta_model = FewShotClassifier(3, 5, 1600).to(device, dtype=torch.double)


from few_shot.models import MatchingNetwork
model3 = MatchingNetwork(1, 5, 15, args.fce, 3,
                        lstm_layers=args.lstm_layers,
                        lstm_input_size=1600,
                        unrolling_steps=args.unrolling_steps,
                        device=device)
from few_shot.models import get_few_shot_encoder
model4 = get_few_shot_encoder(3)
model4.to(device, dtype=torch.double)



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# appn  = get_parameter_number(model1)
maml  = get_parameter_number(model2)
matching  = get_parameter_number(model3)
proto  = get_parameter_number(model4)
print("maml:", maml)
print("matching:", matching)
print("proto:", proto)