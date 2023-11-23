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
from few_shot.proto import proto_net_episode, add_noise
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH, DATA_PATH
from few_shot.metrics import NAMED_METRICS
from utils.utils import *
import json
from torch.optim import Adam
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
parser.add_argument('--noise_type', default="IT")
args = parser.parse_args()

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
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

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_noisetype={args.noise_type}'

# 测试模型
correct = 0
total = 0

###################
# Create datasets #
###################
dataset_class = TrainDataset
test = dataset_class('train', args.dataset)
test_taskloader = DataLoader(
    test,
    batch_sampler=NShotTaskSampler(test, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)

#########
# Model #
#########
checkpoint = torch.load('/home/dsz/cjq/few-shot-master/models/proto_nets/miniImageNet_nt=5_kt=5_qt=5_nv=5_kv=5_qv=5_noisetype=None_train_type=continue_distance=l2.pth')
model = get_few_shot_encoder(num_input_channels)
model.load_state_dict(checkpoint)
model.to(device, dtype=torch.double)

prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train)
fit_function=proto_net_episode
optimiser = Adam(model.parameters(), lr=1e-3)
#########
# Test #
#########
logs = []
seen = 0
metrics=['categorical_accuracy']
totals = {m: 0 for m in metrics}
loss_fn = torch.nn.NLLLoss().cuda()
if loss_fn is not None:
    totals['loss'] = 0
    model.eval()
    with torch.no_grad():
        for batch in test_taskloader:
            x, y, class_name_ = prepare_batch(batch)
            if args.noise_type != None:
                x = add_noise(args.n_train, args.k_train, x, args.noise_type, class_name_, total_class, args.dataset, args.noise_dataset)
            loss, y_pred = fit_function(
                model,
                optimiser,
                loss_fn,
                x,
                y,
                n_shot=args.n_test,
                k_way=args.k_test,
                q_queries=args.q_test,
                train=False,
                distance=args.distance
                # **self.kwargs
            )
            seen += x.shape[0]
            if loss_fn is not None:
                totals['loss'] += loss_fn(y_pred, y).item() * x.shape[0]
            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    # Assume metric is a callable function
                    v = m(y, y_pred)

                totals[m] += v * x.shape[0]

for m in ['loss'] + metrics:
    logs.append(totals[m] / seen)

print(logs)




