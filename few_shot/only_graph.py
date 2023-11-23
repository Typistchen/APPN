import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from few_shot.backbone import CNNEncoder, ResNet12, MultiHeadAttention, SelfAttention, RelationNetwork
from sklearn.neighbors import LocalOutlierFactor
import sys

root_path = "/home/dsz/cjq/few-shot-master"
sys.path.append(root_path)

def calculate_distance_sum(arr, index):
    distance_sum = 0
    for i in range(0, len(arr)):
        dist = (
                torch.abs(arr[i] - arr[index]).sum(dim=-1)
            )
        distance_sum += dist
    distance_sum = distance_sum / (len(arr)-1)
    return distance_sum

def group_normalize(lst, group_size):
    normalized_lst = []
    num_groups = len(lst) // group_size
    #print(num_groups)
    for i in range(num_groups):
        group = lst[i * group_size : (i + 1) * group_size]
        group = torch.tensor(group)
        normalized_group = group / torch.sum(group)
        normalized_lst.extend(normalized_group)
    return normalized_lst

def calculate_weighted_proto(scale, embedings, n_way):
    emb_proto = torch.empty(n_way, embedings.shape[-1])
    scale = scale.reshape(n_way,-1)
    scale = scale.cuda()
    for way in range(n_way) :
        result = scale[way].reshape(-1,1) * embedings[way]
        result = torch.sum(result, dim=0).reshape(1, -1)
        emb_proto[way] = result
    return emb_proto

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
    
def Adjacency_matrix(embedings, metric_method, k):
    W = pairwise_distances(embedings, embedings, metric_method)       
    W = torch.exp(-W/2) # 求指数

    topk, indices = torch.topk(W, k) # 80行里面取出20个最大的数的下标，作为邻接矩阵
    mask = torch.zeros_like(W)  # 初始化全是0的mask
    mask = mask.scatter(1, indices, 1) # 用1去占
    mask = ((mask+torch.t(mask))>0).type(torch.float32)      # union, kNN graph #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
    W = W * mask # 算出真正的邻接矩阵
    return W

def Normalize_Adjacency(W, N, eps):
    D = W.sum(0)
    D_sqrt_inv = torch.sqrt(1.0/(D+eps))
    D1 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
    D2 = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
    S  = D1 * W * D2 # 算出归一化的矩阵
    return S

def Normalize_confidence(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)   
    return normalized_data 

def LOF(embeddings):
    # 创建 LOF 模型
    embeddings_cpu = embeddings.cpu()
    lof = LocalOutlierFactor(n_neighbors=3)
    lof.fit(embeddings_cpu.detach().numpy())
    # 训练模型并预测异常得分
    # y_pred = lof.fit_predict(embeddings_cpu)
    # 获取异常得分

    scores = np.abs(lof.negative_outlier_factor_)
    scores = (torch.tensor(scores)).cuda()
    
    return scores

def Remove_correct_label(index, score):
    result = []
    for i in index:
        if score[i] > 1.5:
            result.append(i)
    result = torch.tensor(result)
    return result

def remove_elements(tensor, indices):
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


class Soft_pseudo_labels(nn.Module):
    # make soft pseudo labels and confidence labels
    def __init__(self, args):
        super(Soft_pseudo_labels, self).__init__()
        self.args = args
        self.n_way = args.n_way_train
        self.k_shot = args.k_shot_train
        self.num_support = self.n_way * self.k_shot
        self.num_query = args.q_train
 
        self.proto_method = args.proto_method
        self.metric_method = args.metric_method

        self.encoder = CNNEncoder(args)
        self.fusion = nn.Conv2d(2, 1, kernel_size=(1,1), stride=(1,1)) #定义fusion层

        self.soft_label_layer = nn.Linear(1600, self.n_way) # 出软标签
        self.confidence_layer = nn.Linear(1600, 1) # 出置信度

    def forward(self, inputs):
        """
        inputs:
        support:    (N_way*N_shot)x3x84x84
        query:      (N_way*N_query)x3x84x84

        s_labels:   (N_way*N_shot)xN_way, one-hot
        q_labels:   (N_way*N_query)xN_way, one-hot
        """
        num_support = self.num_support
        
        embedings = self.encoder(inputs) # 总的embeding个数

        # ebmd_support = embedings[0:num_support, : ] # support的个数
        # scale = []

        # ebmd_support = ebmd_support.view(-1)
        # ebmd_support = ebmd_support.reshape(self.n_way, self.k_shot, -1)


        # ########
        # #计算比例
        # ########
        # if self.proto_method == 'cosine':
        #     for way in range(self.n_way):
        #         temp = ebmd_support[way]
        #         for shot in range(self.k_shot):
        #             distance_sum = calculate_distance_sum(temp, shot)
        #             scale.append(distance_sum)
                
        # scale = group_normalize(scale, self.k_shot)
        # scale = torch.tensor(scale)
        # scale = scale.cuda()
        # #############
        # #计算支持集原型
        # #############
        # emb_proto = calculate_weighted_proto(scale, embedings, self.n_way)    # 原型的大小为[way,shape的最后一个]
        # emb_proto = emb_proto.cuda()

        # ##############
        # #计算每个特征向量和原型的距离
        # #############
        # emb_dis = pairwise_distances(embedings, emb_proto, self.args.metric_method)
        # emd_dis_softmax = nn.Softmax(dim=-1)(emb_dis)

        # ##############
        # #计算距离得分
        # ############
        # dis_score = torch.matmul(emd_dis_softmax, emd_dis_softmax.T)

        # ############
        # #计算 attention 矩阵
        # ############

        # #attention_score = self.attention(inputs)  ##具体大小推算一下
        # attention_emb = embedings.unsqueeze(0)
        # attention_net = SelfAttention(embedings.shape[-1], self.args.hidden_size).cuda() # 目前用单头 
        # #attention_net = MultiHeadAttention(embedings.shape[-1], self.args.hidden_size, self.args.num_heads).cuda()
        
        # attention_score = attention_net(attention_emb)
        # #############n
        # #进行fusion操作
        # #############
        # # torch.cat((tensor1.unsqueeze(0), tensor2.unsqueeze(0)), dim=0)
        # fusion_score = self.fusion(torch.cat((dis_score.unsqueeze(0), attention_score.unsqueeze(0)), dim=0)).squeeze(dim=0)

        # ###############
        # #计算fusion后的矩阵
        # ##############
        # new_embs =  torch.matmul(fusion_score, embedings)

        # soft_pseudo_labels = nn.Softmax(dim=-1)(self.soft_label_layer(new_embs))
        # confifence = nn.Softmax(dim=-1)(self.confidence_layer(new_embs))

        # confifence = torch.sigmoid(confifence) #归一化

        return embedings
    
class   Graph(nn.Module):
    # 建图并进行分割
    def __init__(self, args):
        super(Graph, self).__init__()
        self.args = args

        self.args = args
        self.n_way = args.n_way_train
        self.k_shot = args.k_shot_train
        self.num_support = self.n_way * self.k_shot
        self.num_query = args.q_train
 
        self.proto_method = args.proto_method
        self.metric_method = args.metric_method

        self.soft_pseudo_labels = Soft_pseudo_labels(self.args)
        self.sigma_clulate1 = RelationNetwork()
        self.sigma_clulate2 = RelationNetwork()
        if   self.args.alpha_learn:     # learned sigma, fixed alpha
            self.alpha = torch.tensor([self.args.alpha], requires_grad=False).cuda(0)
        else :          # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([self.args.alpha]).cuda(0), requires_grad=True)

    def forward(self, inputs): 
        eps = np.finfo(float).eps

        embedings = self.soft_pseudo_labels(inputs)

# 获取张量中不重复的类别总数
        num_classes = len(torch.unique(support_labels))

# 使用torch.eye创建一个单位矩阵，其大小为(num_classes, num_classes)
# 然后用索引操作将对应的行提取出来构成one-hot编码
        one_hot_support = torch.eye(num_classes)[support_labels].cuda()
        query_label = torch.eye(num_classes)[query_label].cuda()
        yu = torch.zeros(num_classes*self.num_query, num_classes).cuda(0)
        label_all  = torch.cat((one_hot_support,yu),0).double() # 前5个是support的label 后面是query的label

        ebmd_support = embedings[0:self.num_support, : ] # support的个数

        ##########
        # adjacency matrix parameters
        ##########

        self.sigma1   = self.sigma_clulate1(embedings)
        
        N, d    = embedings.shape[0], embedings.shape[1] # N是节点个数, d是维数
        embedings_adjacency = embedings / (self.sigma1+eps) # N*d  计算所有的 emb_all / sigma [80 ,1600]
        emb_all = embedings_adjacency / (self.sigma1+eps) # N*d  计算所有的 emb_all / sigma [80 ,1600]
        emb1    = torch.unsqueeze(emb_all,1) # N*1*d 插入维度 [80, 1, 1600]
        emb2    = torch.unsqueeze(emb_all,0) # 1*N*d [1, 80, 1600] 

        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N [80 80] 距离的平方再求均值，得到w矩阵的第一版
        W       = torch.exp(-W/2) # 求指数

        topk, indices = torch.topk(W, self.args.alpha_k)
        mask = torch.zeros_like(W)  # 初始化全是0的mask
        mask = mask.scatter(1, indices, 1) # 用1去占
        mask = ((mask+torch.t(mask))>0).type(torch.float32)      # union, kNN graph
            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
        W    = W*mask # 算出真正的邻接矩阵

        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2 # 算出归一化的矩阵

        # ys = s_labels # 只有support的标签
        # yu = torch.zeros(num_classes*num_queries, num_classes).cuda(0) # [75, 5]query的label
        #yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        # y  = torch.cat((ys,yu),0) # 前5个是support的label 后面是query的label
        F  = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha*S+eps), label_all)
        Fq = F[self.num_support:, :] # query predictions

        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((one_hot_support, query_label), 0), 1)
        loss = ce(F, gt)

        predq = torch.argmax(Fq,1)
        gtq   = torch.argmax(query_label,1)
        correct = (predq==gtq).sum()
        total   = self.num_query * num_classes
        acc = 1.0 * correct.float() / float(total)


        ##########
        # calculate adjacency matrix 
        ##########
        # adjacency = Adjacency_matrix(embedings_adjacency, self.args.metric_method, self.args.k)

        #########
        # normolize adjacency matrix 
        ########
        # normalize_adjacency = Normalize_Adjacency(adjacency, N, eps)

        

        ###################
        # LABEL PROPAGATION
        ###################
        # #for times in range(self.args.alpha_k):
        #     Temp  = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha * normalize_adjacency + eps), label_all)
        #     label_all = Temp

        # ##########
        # # remove correct
        # ##########

        # ### 取出标签
        # label_new = label_all[:, :3]
        # confidence_new = label_all[:, 0].unsqueeze(1)

        # ### 归一化
        # normalize_label = F.softmax(label_new, dim=1)
        # normalize_confidence = Normalize_confidence(confidence_new).squeeze()

        # max_score = (torch.max(normalize_label, dim=1)[0])

        # ### 符合条件一的下标
        # correct_indices = torch.nonzero(torch.logical_and(normalize_confidence > 0.2, max_score > 0.2))
        # #correct_indices = torch.nonzero((normalize_confidence > 0.1) & (max_score > 0.1), as_tuple=False)

        # ### 符合条件二
        # correct_indices = correct_indices[correct_indices <= self.num_support]

        # ### 符合条件三
        # LOF_score = LOF(embedings)
        # correct_indices = Remove_correct_label(index=correct_indices, score=LOF_score).long().cuda()

        
        # #计算新图
        # if torch.numel(correct_indices) == 0:
        #     embedings_clean = embedings
        #     soft_pseudo_labels_clean = soft_pseudo_labels
        #     confifence_clean = confifence
        # else:
        #     embedings_clean = remove_elements(embedings,correct_indices)
        #     soft_pseudo_labels_clean = remove_elements(soft_pseudo_labels,correct_indices)
        #     confifence_clean = remove_elements(confifence,correct_indices)


        # self.sigma2   = self.sigma_clulate2(embedings_clean)
        # eps = np.finfo(float).eps
        # N, d    = embedings_clean.shape[0], embedings_clean.shape[1] # N是节点个数, d是维数
        # embedings_adjacency_clean = embedings_clean / (self.sigma2+eps) # N*d  计算所有的 emb_all / sigma [80 ,1600]

        # ##########
        # # calculate adjacency matrix 
        # ##########
        # adjacency_clean = Adjacency_matrix(embedings_adjacency_clean, self.args.metric_method, self.args.k)

        # #########
        # # normolize adjacency matrix 
        # ########
        # normalize_adjacency_clean = Normalize_Adjacency(adjacency_clean, N, eps)

        # label_all_clean = torch.cat((soft_pseudo_labels_clean, confifence_clean), dim=1)

        # ###################
        # # LABEL PROPAGATION
        # ###################
        # for times in range(self.args.alpha_k):
        #     Temp  = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha * normalize_adjacency_clean + eps), label_all_clean)
        #     label_all_clean = Temp

        return  loss, acc

        









