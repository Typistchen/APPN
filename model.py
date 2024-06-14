#-------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2019.1.11
# Author: Yanbin Liu
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from few_shot.gnn import calculate_distance_sum, group_normalize, calculate_weighted_proto, Normalize_confidence,Remove_incorrect_label,LOF,remove_elements 
from few_shot.backbone import SelfAttention, MultiHeadAttention, ResNet12
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
        distances = ((
                x - y
                
        )**2).mean(2)
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
    
def pairwise_distances_pro(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    n_x = x.shape[0]
    n_y = y.shape[0]
    EPSILON = 1e-8
    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
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

class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self, args):
        
        super(CNNEncoder, self).__init__()
        self.num_query = args.n_way * args.k_shot
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
        self.fc1 = nn.Linear(1600, 5)
        self.fc2 = nn.Linear(1600, 2)
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self,x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        embedding = self.layer4(out).view(x.size(0), -1)
        return embedding



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

    def forward(self, x):
        
        x = x.view(-1,64,5,5)
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) # no relu
        out = out.view(out.size(0),-1) # bs*1
        return out
    
class RelationNetworkCifar(nn.Module):
    """Graph Construction Module"""
    def __init__(self):
        super(RelationNetworkCifar, self).__init__()

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

    def forward(self, x):
        
        x = x.view(-1,64,2,2)
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) # no relu
        out = out.view(out.size(0),-1) # bs*1
        return out


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes


class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, args, encoder):
        super(LabelPropagation, self).__init__()

        self.im_width, self.im_height, self.channels = list(map(int, args.x_dim.split(',')))

        self.h_dim, self.z_dim = args.h_dim, args.z_dim
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_query = args.n_query
        self.args = args
        self.encoder = encoder

        self.num_support = args.n_way * args.k_shot
        self.num_all = self.n_way*( self.k_shot+ self.n_query)
        

        self.alpha = nn.Parameter(torch.tensor([args.alpha]).cuda(0), requires_grad=True)

        self.fusion = nn.Conv2d(2, 1, kernel_size=(1,1), stride=(1,1)) 

        self.soft_label_layer = nn.Linear(1600, self.args.n_way) 
        self.slf_attn = MultiHeadAttention(1, 1600, self.num_all, dropout=0) 
        self.relation = RelationNetwork()


        self.loss_fn = torch.nn.NLLLoss().cuda()

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

        [support, s_labels, query, q_labels, noise_idxes] = inputs

        inp   = torch.cat((support,query), 0)

        emb_all = self.encoder(inp)
        
        support_emb = emb_all[:self.n_way*self.k_shot]
        queries_emb = emb_all[self.n_way*self.k_shot:]

        attention_emb = emb_all.unsqueeze(0)
        attention_score = self.slf_attn(attention_emb, attention_emb)
        matrix = torch.zeros(self.n_way*( self.k_shot+ self.n_query), self.n_way*( self.k_shot+ self.n_query))                    
        matrix.fill_diagonal_(1)
        attention_score = attention_score + matrix.cuda()
        new_embs =  torch.matmul(attention_score.view(self.num_all,self.num_all), emb_all) 

        support_emb_new = new_embs[:self.n_way*self.k_shot]
        queries_emb_new = new_embs[self.n_way*self.k_shot:]
        
        support_emb = support_emb_new.reshape(self.n_way, self.k_shot, -1)
        scale = torch.zeros((self.n_way,self.k_shot)).cuda()

        for way in range(self.n_way):
                temp = support_emb[way]
                for shot in range(self.k_shot):
                    distance_sum = calculate_distance_sum(temp, shot)
                    scale[way,shot] = distance_sum
        
    
        scale = (-scale).softmax(dim=1)

        emb_proto = calculate_weighted_proto(scale, support_emb, self.n_way,self.k_shot)    
        
        y_support = s_labels                                       

        distances = pairwise_distances_pro(queries_emb, emb_proto, 'l2')
        y_pred = (-distances).softmax(dim=1)
        label_all  = torch.cat((y_support,y_pred),0)   
        log_p_y = (-distances).log_softmax(dim=1)

        N, d    = new_embs.shape[0], new_embs.shape[1]      

        self.sigma   = self.relation(new_embs)             

        new_embs = new_embs / (self.sigma+eps)              
        W = pairwise_distances(new_embs, new_embs, 'l2')
        W       = torch.exp(-W/2)                           
        
        if self.args.k>0:

            topk, indices = torch.topk(W, self.args.k)      
            mask = torch.zeros_like(W)                     
            mask = mask.scatter(1, indices, 1)              
            mask = ((mask+torch.t(mask))>0).type(torch.float32)      
            W    = W*mask                                  

        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2                                   

        encoded_array = np.zeros((self.num_all*self.num_all, 2))
        encoded_array[:, 0] = 1 

        index = S
        index = index.to("cpu")
        index = index.ravel()
        encoded_array[index > 0, :] = np.array([0, 1])
        encoded_array = (torch.tensor(encoded_array)).cuda()

        label_after_propagate  = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha*S+eps), label_all)



        ### 取出标签

        label_all_after_propagate = label_after_propagate[:, :self.n_way]
        
        normalize_label_all_after_propagate = torch.softmax(label_all_after_propagate, dim=1)

        max_score = (torch.max(normalize_label_all_after_propagate, dim=1)[0])
        # **************************************************************************************************************************    
       
        incorrect_indices = []
        for i in range(100):
            if  max_score[i] < 0.7:
                incorrect_indices.append(i)
        incorrect_indices = torch.tensor(incorrect_indices)
        
        incorrect_indices = incorrect_indices[incorrect_indices < self.num_support]

        LOF_score = LOF(new_embs)
        incorrect_indices = Remove_incorrect_label(index=incorrect_indices, score=LOF_score).long().cuda()

        if torch.numel(incorrect_indices) == 0:

            clean_label_all = normalize_label_all_after_propagate
            s_labels = s_labels

            S = S
            label_all = label_all
        else:

            clean_label_all = remove_elements(normalize_label_all_after_propagate,incorrect_indices)
            label_all = remove_elements(label_all,incorrect_indices)
            s_labels = remove_elements(s_labels,incorrect_indices)

            S = torch.cat((S[:incorrect_indices[0]], S[incorrect_indices[0] + 1:]), dim=0)
            for row_index in incorrect_indices[1:]:
                S = torch.cat((S[:row_index], S[row_index + 1:]), dim=0)

            S = torch.cat((S[:, :incorrect_indices[0]], S[:, incorrect_indices[0] + 1:]), dim=1)
            for col_index in incorrect_indices[1:]:
                S = torch.cat((S[:, :col_index], S[:, col_index + 1:]), dim=1)

        num_incorrect = int(incorrect_indices.shape[0])

        matrix_size = self.num_all
        symmetric_matrix = torch.zeros((matrix_size, matrix_size))
        symmetric_matrix.fill_diagonal_(1)
        for m in range(5):
            for i in range(5):
                for j in range(5):
                    symmetric_matrix[i+m*5, j+m*5] = symmetric_matrix[j+m*5, i+m*5] = 1  
                for x in range(self.n_query):
                    symmetric_matrix[i+m*5, 25 + x +m*self.n_query] = symmetric_matrix[x + 25+m*self.n_query, i+m*5] = 1  
        for t in range(5):
            for p in range(self.n_query):
                for q in range(self.n_query):
                    symmetric_matrix[p+25+t*15, 25+q+t*self.n_query] = symmetric_matrix[ p+25+t*self.n_query,25+q+t*self.n_query] = 1 

        edge = symmetric_matrix.type(torch.float32).cuda() 
        flat_array = edge.ravel().cuda()
        flat_array = flat_array.long()

        N = self.num_all - num_incorrect
        label_after_2_Propagate = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha*S+eps), clean_label_all)

        query_after_2_Propagate = label_after_2_Propagate[self.n_way*self.k_shot - num_incorrect:, :]                           # query predictions
  
        ce = nn.CrossEntropyLoss().cuda(0)
        
        label_rogin = torch.cat((s_labels,q_labels),0)
        gt_all = torch.argmax(label_rogin,1)
        gt_q = torch.argmax(q_labels,1)
 
        loss4 = ce(label_after_2_Propagate, gt_all)
        
        loss2 = ce(encoded_array, flat_array)
        loss =  0.5 * loss4 + 0.5*loss2 
        ## acc
        predq_2 = torch.argmax(query_after_2_Propagate,1)

        gtq   = torch.argmax(q_labels,1)

        correct_2 = (predq_2==gtq).sum()

        total   = self.k_shot * self.n_query

        acc_2 = 1.0 * correct_2.float() / float(total)


        return loss, acc_2
    
