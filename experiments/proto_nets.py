"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import sys
root_path = "/home/dsz/cjq/few-shot-master"

sys.path.append(root_path)

from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from few_shot.datasets import TrainDataset
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train_gcn import proto_net_episode_gcn
from few_shot.gcn import LabelPropagation
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH, DATA_PATH
from utils.utils import *
import json

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='miniImageNet')
parser.add_argument('--noise_dataset',default='cifar')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=5, type=int)
parser.add_argument('--noise_type', default=None)
parser.add_argument('--train_type', default=None)
args = parser.parse_args()

if args.noise_type == "OOT":
    path = DATA_PATH + '/' + args.dataset + '/'+ args.dataset + '_train.json'
    with open(path) as f:
        total_class = json.load(f)

elif args.noise_type == "OOD":
    path = DATA_PATH + '/' + args.noise_dataset + '/'+ args.noise_dataset + '_train.json'
    with open(path) as f:
        total_class = json.load(f)
else:
    total_class = None

evaluation_episodes = 1000
episodes_per_epoch = 100
n_epochs = 80
num_input_channels = 3
drop_lr_every = 40


if args.train_type == None:
    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_noisetype={args.noise_type}_distance={args.distance}'
else:
    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_noisetype={args.noise_type}_train_type={args.train_type}_distance={args.distance}'

print(param_str)

###################
# Create datasets #
###################
if args.dataset not in ['tieredImagenet', 'cifar', 'fc100', 'miniImageNet']:
    raise ValueError('Unsupported dataset')
dataset_class = TrainDataset
train = dataset_class('train', args.dataset)
background_taskloader = DataLoader(
    train,
    batch_sampler=NShotTaskSampler(train, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
evaluation = dataset_class('val', args.dataset)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)

#########
# Model #
#########
if args.train_type == None:
    model = LabelPropagation(args)
    #model = get_few_shot_encoder(num_input_channels)
    model.to(device, dtype=torch.double)

############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    # if epoch % drop_lr_every == 0:
    #     return lr / 2
    # else:
    return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance,
        noise_dataset=args.noise_dataset,
        total_class = total_class,
        dataset = args.dataset,
        noise_type = args.noise_type
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
] # 结合多个要的callbacks

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode_gcn,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance, 'noise_type':args.noise_type, 'total_class':total_class, 
                         'dataset':args.dataset, 'noise_dataset':args.noise_dataset},
)



