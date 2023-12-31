import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
        # h_dim, z_dim = args['h_dim'], args['z_dim']
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self,x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out 


class RelationNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2*2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)            # max-pool without padding 
        self.m1 = nn.MaxPool2d(2, padding=1) # max-pool with padding

    def forward(self, x, rn):
        
        x = x.view(-1,64,5,5)
        
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) # no relu

        out = out.view(out.size(0),-1) # bs*1

        return out

a = 1
class Prototypical(nn.Module):
    """Main Module for prototypical networlks"""
    def __init__(self, args):
        super(Prototypical, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.encoder = CNNEncoder(args)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        inp   = torch.cat((support,query), 0)
        emb   = self.encoder(inp) # 80x64x5x5
        emb_s, emb_q = torch.split(emb, [num_classes*num_support, num_classes*num_queries], 0)
        emb_s = emb_s.view(num_classes, num_support, 1600).mean(1)
        emb_q = emb_q.view(-1, 1600)
        emb_s = torch.unsqueeze(emb_s,0)     # 1xNxD
        emb_q = torch.unsqueeze(emb_q,1)     # Nx1xD
        dist  = ((emb_q-emb_s)**2).mean(2)   # NxNxD -> NxN

        ce = nn.CrossEntropyLoss().cuda(0)
        loss = ce(-dist, torch.argmax(q_labels,1))
        ## acc
        pred = torch.argmax(-dist,1)
        gt   = torch.argmax(q_labels,1)
        correct = (pred==gt).sum()
        total   = num_queries*num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc


class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, args):
        super(LabelPropagation, self).__init__()
        # self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        # self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.encoder = CNNEncoder(args)
        self.relation = RelationNetwork()

        #if   args['rn'] == 300:   # learned sigma, fixed alpha
        self.alpha = torch.tensor(0.99, requires_grad=False).cuda(0)
        #elif args['rn'] == 30:    # learned sigma, learned alpha
        #self.alpha = nn.Parameter(torch.tensor([args['alpha']]).cuda(0), requires_grad=True)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = np.finfo(float).eps

        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1] # num_classes = 5
        num_support = int(s_labels.shape[0] / num_classes) # num_support = 5
        num_queries = int(query.shape[0] / num_classes) # num_queries = 1

        # Step1: Embedding
        inp     = torch.cat((support,query), 0) # 把query 和 support 合并起来
        #emb_all = self.encoder(inp).view(-1,1600)
        emb_all = self.encoder(inp).reshape(-1,1600) # 【80， 1600】
        N, d    = emb_all.shape[0], emb_all.shape[1] # N是节点个数, d是维数

        # Step2: Graph Construction
        ## sigmma
        # if self.args['rn'] in [30,300]:
        self.sigma   = self.relation(emb_all, self.args['rn'])  # 用网络计算sigma [80, 1]
            
            ## W
        emb_all = emb_all / (self.sigma+eps) # N*d  计算所有的 emb_all / sigma [80 ,1600]
        emb1    = torch.unsqueeze(emb_all,1) # N*1*d 插入维度 [80, 1, 1600]
        emb2    = torch.unsqueeze(emb_all,0) # 1*N*d [1, 80, 1600] 
            # [2, 1, 2] - [1, 2, 2]可以相减
            # 2个2维的减2个2维的
            # [80, 1, 1600] - [1, 80, 1600]
            # 80个1600维 减 80个1600维
            # 前的第一个减后面的80个 -> 前面第二个减后面80个
        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N [80 80] 距离的平方再求均值，得到w矩阵的第一版
        W       = torch.exp(-W/2) # 求指数

        ## keep top-k values
        if self.args['k']>0:
            topk, indices = torch.topk(W, self.args['k']) # 80行里面取出20个最大的数的下标，作为邻接矩阵
            mask = torch.zeros_like(W)  # 初始化全是0的mask
            mask = mask.scatter(1, indices, 1) # 用1去占
            mask = ((mask+torch.t(mask))>0).type(torch.float32)      # union, kNN graph
            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W    = W*mask # 算出真正的邻接矩阵

        ## normalize
        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2 # 算出归一化的矩阵

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels # 只有support的标签
        yu = torch.zeros(num_classes*num_queries, num_classes).cuda(0) # [75, 5]query的label
        #yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y  = torch.cat((ys,yu),0) # 前5个是support的label 后面是query的label
        F  = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha*S+eps), y)
        Fq = F[num_classes*num_support:, :] # query predictions
        
        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F, gt)
        ## acc
        predq = torch.argmax(Fq,1)
        gtq   = torch.argmax(q_labels,1)
        correct = (predq==gtq).sum()
        total   = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc