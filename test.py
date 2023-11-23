import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from few_shot.backbone import CNNEncoder, ResNet12
from few_shot.gnn import Adjacency_matrix, Normalize_Adjacency
test_instance = [[1,1],[2,2],[3, 3]],[[3,3],[4,4], [6, 6]]
test_instance = torch.tensor(test_instance)



a = test_instance[0]


def calculate_distance_sum(arr, index):
    distance_sum = 0
    for i in range(0, len(arr)):
        dist = (
                torch.abs(arr[i] - arr[index]).sum(dim=-1)
            )
        distance_sum += dist
    distance_sum = distance_sum / (len(arr)-1)
    return distance_sum
scale=[]
print(len(a))
for j in range(test_instance.shape[0]):
    a = test_instance[j]
    for i in range(len(a)):
        distance_sum = calculate_distance_sum(a, i)
        scale.append(distance_sum)
print(scale)
def group_normalize(lst, group_size):
    normalized_lst = []
    num_groups = len(lst) // group_size
    print(num_groups)
    for i in range(num_groups):
        group = lst[i * group_size : (i + 1) * group_size]
        normalized_group = group / np.sum(group)
        normalized_lst.extend(normalized_group)

    return normalized_lst
scale = group_normalize(scale, 3)




# 创建两个示例张量
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([5, 6])

# 使用广播对应位置相乘
result = tensor1 * tensor2.reshape(-1, 1)

print("result:",result)

emb = torch.empty((2,test_instance.shape[-1]))
print(emb.shape)
scale = torch.tensor(scale).reshape(2,-1)
print(scale.shape)
for way in range(2) :
    print(scale[way])
    print(test_instance[way])

    result = scale[way].reshape(-1,1) * test_instance[way]

    print(result)
# 相加
    result = torch.sum(result, dim=0).reshape(1, -1)
    emb[way] = result
   

def calculate_weighted_proto(scale, embedings, n_way):
    emb_proto = torch.empty(n_way, embedings.shape[-1])
    scale = torch.tensor(scale).reshape(n_way,-1)
    for way in range(n_way) :
        result = scale[way].reshape(-1,1) * embedings[way]
        result = torch.sum(result, dim=0).reshape(1, -1)
        emb_proto[way] = result
    return emb_proto

emd_proto = calculate_weighted_proto(scale, test_instance, 2)
print(emd_proto)


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    EPSILON = 1e-8
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        x = x.expand(n_x, n_y, -1)
        y = y.expand(n_x, n_y, -1)
        distances = (
                x - y
                
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))
    


# import torch

# test_instance = [[1,1], [1,2], [1,3], [2,1], [2,2],[2,3], [3,1], [3,2], [3,3],
#                  [101,101], [101,102], [101,103], [102,101], [102,102],[102,103], [103,101], [103,102], [103,103],[500,500]]

# label = [[1,0,1],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
#          [0.7,0,3],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0.5,0.5,1]]
# test_instance = torch.tensor(test_instance)
# label = torch.tensor(label)
# label = label.cuda()
# label = label.float()
# W = pairwise_distances(test_instance, test_instance, "l2")       
# W = torch.exp(-W/2) # 求指数

# topk, indices = torch.topk(W, 8) # 80行里面取出20个最大的数的下标，作为邻接矩阵
# mask = torch.zeros_like(W)  # 初始化全是0的mask
# mask = mask.scatter(1, indices, 1) # 用1去占
# mask = ((mask+torch.t(mask))>0).type(torch.float32)      # union, kNN graph #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
# W = W * mask # 算出真正的邻接矩阵

# eps = np.finfo(float).eps
# n = test_instance.shape[0]
# normalize_adjacency = Normalize_Adjacency(W, n, eps)
# normalize_adjacency = normalize_adjacency.cuda()

# for times in range(10):
#     F  = torch.matmul(torch.inverse(torch.eye(n).cuda(0)-0.2 * normalize_adjacency + eps), label)
#     label = F


# print(W)
# print(normalize_adjacency)
# print(label)
    
import torch

# 原始数据
data = torch.randn(5, 3)
print("原始数据形状：", data)

# 每3个元素取最大值
max_values = torch.max(data, dim=1)[0]
result = max_values.unsqueeze(1)
print("处理后的数据形状：", result.shape)
print("处理后的数据：\n", result)

import torch

# 张量a和b
a = torch.tensor([[1], [3], [4], [2], [5]])
b = torch.tensor([[2], [4], [5], [1], [6]])

# 找到满足条件的下标
indices = torch.nonzero((a > 2) & (b > 3), as_tuple=False)

# 打印结果
print("满足条件的下标：\n", indices)

import torch
import torch.nn as nn

# 创建对数概率和目标标签
log_probs = torch.tensor([[-0.5, -1.2, -2.1],  # 对数概率值，假设为模型的输出
                         [-1.1, -0.6, -1.8],
                         [-0.9, -1.3, -0.4]])
target = torch.tensor([0, 2, 1])  # 目标标签

# 将对数概率和目标标签移到CUDA设备上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_probs = log_probs.to(device)
target = target.to(device)

# 创建NLLLoss损失函数实例
loss_fn = nn.NLLLoss().to(device)

# 计算损失
loss = loss_fn(log_probs, target)
print(loss.item())
