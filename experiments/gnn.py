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
<<<<<<< HEAD
# import torchsummary as summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
=======

>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()

parser.add_argument('--distance', default='cosine')

#attention
parser.add_argument('--num_heads', default=2)
parser.add_argument('--hidden_size', default=128)
parser.add_argument('--hidden_dropout_prob', default=0.2)

parser.add_argument('--proto_method', default="cosine")
parser.add_argument('--metric_method', default="cosine")

#graph
# parser.add_argument('--k', default=10)
# parser.add_argument('--alpha_learn', default=False)
# parser.add_argument('--alpha', default=0.2)
# parser.add_argument('--alpha_k', default=20)
# parser.add_argument('--encoder_name', default="CNNEncoder")



###############
<<<<<<< HEAD
parser.add_argument('--gpu',        type=str,   default=1,          metavar='GPU',
=======
parser.add_argument('--gpu',        type=str,   default=0,          metavar='GPU',
>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c
                    help="gpus, default:0")
# model params
n_examples = 600
parser.add_argument('--x_dim',      type=str,   default="84,84,3",  metavar='XDIM',
                    help='input image dims')
parser.add_argument('--h_dim',      type=int,   default=64,         metavar='HDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--z_dim',      type=int,   default=64,         metavar='ZDIM',
                    help="dimensionality of output channels (default: 64)")

# training hyper-parameters
n_episodes = 100 # test interval
parser.add_argument('--n_way',      type=int,   default=5,          metavar='NWAY',
                    help="nway")
parser.add_argument('--k_shot',     type=int,   default=5,          metavar='NSHOT',
                    help="kshot")
parser.add_argument('--n_query',    type=int,   default=15,         metavar='NQUERY',
                    help="nquery")
parser.add_argument('--n_epochs',   type=int,   default=2100,       metavar='NEPOCHS',
                    help="nepochs")
<<<<<<< HEAD
parser.add_argument('--scale',   type=int,   default=1,       metavar='NEPOCHS',
                    help="nepochs")
=======
>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c
# test hyper-parameters
parser.add_argument('--n_test_way', type=int,   default=5,          metavar='NTESTWAY',
                    help="ntestway")
parser.add_argument('--k_test_shot',type=int,   default=5,          metavar='NTESTSHOT',
                    help="ntestshot")
parser.add_argument('--n_test_query',type=int,  default=15,         metavar='NTESTQUERY',
                    help="ntestquery")

# dataset params
parser.add_argument('--dataset',    type=str,   default='cifar',  metavar='DATASET',
                    help="mini or tiered")
parser.add_argument('--noise_dataset',  type=str,   default='cifar',     metavar='NOISEDATASET',
                    help="mini or tiered")
parser.add_argument('--noise_type',  type=str,   default='IT',     metavar='NOISETYPE',
                    help="mini or tiered")


# label propagation params
parser.add_argument('--alg',        type=str,   default='TPN',      metavar='ALG',
                    help="algorithm used, TPN")
parser.add_argument('--k',          type=int,   default=21,         metavar='K',
                    help="top k in constructing the graph W")
parser.add_argument('--sigma',      type=float, default=0.25,       metavar='SIGMA',
                    help="Initial sigma in label propagation")
parser.add_argument('--alpha',      type=float, default=0.99,       metavar='ALPHA',
                    help="Initial alpha in label propagation")
parser.add_argument('--rn',         type=int,   default=300,        metavar='RN',
                    help="graph construction types: "
                    "300: sigma is learned, alpha is fixed" +
                    "30:  both sigma and alpha learned")

# save and restore params
parser.add_argument('--seed',       type=int,   default=1000,       metavar='SEED',
                    help="random seed for code and data sample")
parser.add_argument('--exp_name',   type=str,   default='exp',      metavar='EXPNAME',
                    help="experiment name")
parser.add_argument('--iters',      type=int,   default=0,          metavar='ITERS',
                    help="iteration to restore params")



###################
# Create datasets #
###################


args = parser.parse_args()
print(args)

if args.noise_type == "OOT":
    train_path = DATA_PATH + '/' + args.dataset + '/'+ args.dataset + '_train.json'
    with open(train_path) as f:
        train_total_class = json.load(f)
    val_path = DATA_PATH + '/' + args.dataset + '/'+ args.dataset + '_val.json'
    with open(val_path) as f:
        val_total_class = json.load(f)

elif args.noise_type == "OOD":
    train_path = DATA_PATH + '/' + args.noise_dataset + '/'+ args.noise_dataset + '_train.json'
    with open(train_path) as f:
        train_total_class = json.load(f)
    val_path = DATA_PATH + '/' + args.noise_dataset + '/'+ args.noise_dataset + '_val.json'
    with open(val_path) as f:
        val_total_class = json.load(f)
<<<<<<< HEAD
elif args.noise_type == "mixed":
    train_path = DATA_PATH + '/' + args.noise_dataset + '/'+ args.noise_dataset + '_train.json'
    with open(train_path) as f:
        train_total_class_ood = json.load(f)
    val_path = DATA_PATH + '/' + args.noise_dataset + '/'+ args.noise_dataset + '_val.json'
    with open(val_path) as f:
        val_total_class_ood = json.load(f)

    train_path = DATA_PATH + '/' + args.dataset + '/'+ args.dataset + '_train.json'
    with open(train_path) as f:
        train_total_class_oot = json.load(f)
    val_path = DATA_PATH + '/' + args.dataset + '/'+ args.dataset + '_val.json'
    with open(val_path) as f:
        val_total_class_oot = json.load(f)
else: 
    train_total_class = None
    val_total_class = None

=======
else:
    train_total_class = None
    val_total_class = None


>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c
if args.dataset not in ['tieredImagenet', 'cifar', 'fc100', 'miniImageNet']:
    raise ValueError('Unsupported dataset')

episodes_per_epoch = 200

dataset_class = TrainDataset
train = dataset_class('train', args.dataset) 
train_taskloader = DataLoader(
    train,
    batch_sampler=NShotTaskSampler(train, episodes_per_epoch, args.k_shot, args.n_way, args.n_query),
    num_workers=4
)
evaluation = dataset_class('val', args.dataset)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.k_shot, args.n_way, args.n_query),
    num_workers=4
)

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
backbone = CNNEncoder(args)
model = model.LabelPropagation(args, backbone)
<<<<<<< HEAD

model.cuda()  
=======
model.cuda(0)  
>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c


model_optim = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=0.02)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) 
scheduler = torch.optim.lr_scheduler.StepLR(model_optim,step_size=100,gamma = 0.5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=100,  eta_min=1e-6)
best_acc = 0
for ep in range(num_epochs):

    loss_tr = []
    ce_list = []
    acc_tr  = []
    loss_val= []
    acc_val = []
    

    for batch_index, batch in tqdm((enumerate(train_taskloader)), desc='train_epoc:{}'.format(ep)):
        prepare_batch = prepare_nshot_task(5, 5, 15)

        x, query_y, support_y, class_name = prepare_batch(batch)
        # noise_idxes = 0
<<<<<<< HEAD
        
        # x,noise_idxes = add_noise(5, 5, x, args.noise_type, class_name, train_total_class, args.dataset, args.noise_dataset, "train")
        if args.noise_type == 'mixed': 
            x,noise_idxes = add_noise(5, 5, x, "OOD", class_name, train_total_class_ood, args.dataset, args.noise_dataset, "train", args.scale)
            x,noise_idxes = add_noise(5, 5, x, "IT", class_name, train_total_class_oot, args.dataset, args.noise_dataset, "train",args.scale)
        elif args.noise_type == 'normal' :
            pass
        else:
            x,noise_idxes = add_noise(5, 5, x, args.noise_type, class_name, train_total_class, args.dataset, args.noise_dataset, "train",args.scale)
=======
        x,noise_idxes = add_noise(5, 5, x, args.noise_type, class_name, train_total_class, args.dataset, args.noise_dataset, "train")
>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c
        support_y = support_y.to('cpu')
        query_y = query_y.to('cpu')
        x = x.float()
        support_a = x[:25,:,:,:]
        query_a = x[25:,:,:,:]
        one_hot_support = torch.eye(5)[support_y].cuda()
        one_hot_query = torch.eye(5)[query_y].cuda()

        model.train()

<<<<<<< HEAD
        inputs = [support_a, one_hot_support.cuda(), query_a, one_hot_query.cuda(),noise_idxes]
        # parameter = model.named_parameters
=======
        inputs = [support_a, one_hot_support.cuda(0), query_a, one_hot_query.cuda(0),noise_idxes]
>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c

        loss, acc = model(inputs)

        loss_tr.append(loss.item())
        acc_tr.append(acc.item())

        model.zero_grad()
        loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 4.0)
        model_optim.step()

        
    for batch_index, batch in tqdm((enumerate(evaluation_taskloader)), desc='test_epoc:{}'.format(ep)):
            # set eval mode
        model.eval()
        prepare_batch = prepare_nshot_task(5, 5, 15)
        x, query_y, support_y, class_name = prepare_batch(batch)
<<<<<<< HEAD
        noise_idxes = []
        # x, = add_noise(5, 5, x, args.noise_type, class_name, val_total_class, args.dataset, args.noise_dataset,"val" )
        if args.noise_type == 'mixed': 
            x,noise_idxes = add_noise(5, 5, x, "OOD", class_name, val_total_class_ood, args.dataset, args.noise_dataset, "val",args.scale)
            x,noise_idxes = add_noise(5, 5, x, "IT", class_name, val_total_class_oot, args.dataset, args.noise_dataset, "val",args.scale)
        else:
            x,noise_idxes = add_noise(5, 5, x, args.noise_type, class_name, val_total_class, args.dataset, args.noise_dataset, "val",args.scale)
        

=======
        
        x,noise_idxes = add_noise(5, 5, x, args.noise_type, class_name, val_total_class, args.dataset, args.noise_dataset,"val" )
>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c
        x = x.float()
        support_a = x[:25,:,:,:]
        query_a = x[25:,:,:,:]
        support_y = support_y.to('cpu')
        query_y = query_y.to('cpu')
<<<<<<< HEAD
        one_hot_support = torch.eye(5)[support_y].cuda(1)
        one_hot_query = torch.eye(5)[query_y].cuda(1)
=======
        one_hot_support = torch.eye(5)[support_y].cuda()
        one_hot_query = torch.eye(5)[query_y].cuda()
>>>>>>> a52eedf06cdbf6397495d8911024768249f9391c
    
        with torch.no_grad():
            inputs = [support_a.cuda(0), one_hot_support.cuda(0), query_a.cuda(0), one_hot_query.cuda(0), noise_idxes]
            loss, acc = model(inputs)
            

        loss_val.append(loss.item() )
        acc_val.append(acc.item())

    if np.mean(acc_val) > best_acc:
        best_acc = np.mean(acc_val)
        path = "40" + args.noise_type + "ture" + args.dataset + args.noise_dataset + '.pth'
        torch.save(model.state_dict(), path)
    print('epoch:{}, loss_tr:{:.5f}, acc_tr:{:.5f},loss_val:{:.5f}, acc_val:{:.5f}, best_acc:{:.5}'.format(ep, np.mean(loss_tr), np.mean(acc_tr), np.mean(loss_val), np.mean(acc_val),best_acc))
    result = [("bili", "40%"+"ture"),("noise_type", args.noise_type+"ture"),("dataset", args.dataset),("noise_dataset", args.noise_dataset),("lr",lr),("backbone",type(backbone).__name__),("epoch",ep),("loss_tr", np.mean(loss_tr)), ("acc_tr", np.mean(acc_tr)),("loss_val", np.mean(loss_val)),("acc_val", np.mean(acc_val)),("best:",best_acc)]
    path = "result.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(result)
    scheduler.step()

 