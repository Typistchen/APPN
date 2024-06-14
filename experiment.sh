
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset cifar               --noise_type IT     --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset fc100               --noise_type IT     --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset tieredImagenet      --noise_type IT     --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset miniImageNet        --noise_type IT     --scale 1

python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset cifar               --noise_type IT     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset fc100               --noise_type IT     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset tieredImagenet      --noise_type IT     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset miniImageNet        --noise_type IT     --scale 2

python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset cifar               --noise_type OOT    --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset fc100               --noise_type OOT    --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset tieredImagenet      --noise_type OOT    --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset miniImageNet        --noise_type OOT    --scale 1

python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset cifar               --noise_type OOT     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset fc100               --noise_type OOT     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset tieredImagenet      --noise_type OOT     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset miniImageNet        --noise_type OOT     --scale 2

python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset cifar               --noise_dataset miniImageNet --noise_type OOD     --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset fc100               --noise_dataset miniImageNet --noise_type OOD      --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset tieredImagenet      --noise_dataset cifar --noise_type OOD     --scale 1
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset miniImageNet        --noise_dataset cifar --noise_type OOD     --scale 1

python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset cifar               --noise_dataset miniImageNet --noise_type OOD     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset fc100               --noise_dataset miniImageNet --noise_type OOD      --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset tieredImagenet      --noise_dataset cifar --noise_type OOD     --scale 2
python  /home/dsz/Documents/cjq/few-shot-master/experiments/gnn.py    --dataset miniImageNet        --noise_dataset cifar --noise_type OOD     --scale 2

python  /home/dsz/Documents/cjq/TPN/experiments/gnn.py    --dataset cifar               --noise_dataset miniImageNet --noise_type mixed     --scale 1
python  /home/dsz/Documents/cjq/TPN/experiments/gnn.py    --dataset fc100               --noise_dataset miniImageNet --noise_type mixed      --scale 1
python  /home/dsz/Documents/cjq/TPN/experiments/gnn.py    --dataset tieredImagenet      --noise_dataset cifar --noise_type mixed     --scale 1
python  /home/dsz/Documents/cjq/TPN/experiments/gnn.py    --dataset miniImageNet        --noise_dataset cifar --noise_type mixed     --scale 1