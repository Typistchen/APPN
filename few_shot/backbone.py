import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

class CNNEncoder(nn.Module):  
    """Encoder for feature embedding 6E4"""
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
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
        # out = out.view(out.size(0), -1)
        return out
         
    
class ResNet12Block(nn.Module):
    """
    ResNet Block
    resnet 的 组件
    """
    def __init__(self, inplanes, planes):
        super(ResNet12Block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        out.view(out.size(0), -1)
        return out
    
class ResNet12(nn.Module):
    """
    ResNet12 Backbone
    """
    def __init__(self, emb_size, block=ResNet12Block, cifar_flag=False):
        super(ResNet12, self).__init__()
        cfg = [64, 128, 256, 512]
        # layers = [1, 1, 1, 1]
        iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU()
        self.emb_size = emb_size
        self.layer1 = self._make_layer(block, cfg[0], cfg[0])
        self.layer2 = self._make_layer(block, cfg[0], cfg[1])
        self.layer3 = self._make_layer(block, cfg[1], cfg[2])
        self.layer4 = self._make_layer(block, cfg[2], cfg[3])
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AvgPool2d(6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        self.layer_second = nn.Sequential(nn.Linear(in_features=layer_second_in_feat,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes):
        layers = []
        layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)           # 3 -> 64
        x = self.bn1(x)             # 64 -> 64
        x = self.relu(x)
        x = self.layer1(x)          # 64 -> 128
        x = self.layer2(x)          # 128 -> 256
        inter = self.layer3(x)      # 256 -> 512
        x = self.layer4(inter)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # 512 -> 128
        x = self.layer_last(x)
        inter = self.maxpool(inter)
        # 256 * 5 * 5
        inter = inter.view(inter.size(0), -1)
        # 256 * 5 * 5 -> 128
        inter = self.layer_second(inter)
        out = []
        out.append(x)
        out.append(inter)
        # no FC here
        return out


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        attention 的 组件
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    

class Attention(nn.Module):
    """
    attention backbone

    """
    def __init__(self, args):
        """
        num_attention_heads 注意力的头的个数
        input_size 输入图片的大小
        hidden_size 特征维度的大小
        hidden_dropout_prob 隐藏去除探针？？
        """
        super(Attention, self).__init__()

        attention_probs_dropout_prob = 0.2

        self.args = args
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads) # 每个头的特征大小
        self.all_head_size = args.hidden_size # 所有头的大小的组合

        self.query = nn.Linear(args.input_size, self.all_head_size)
        self.key = nn.Linear(args.input_size, self.all_head_size)
        self.value = nn.Linear(args.input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        #value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)

        # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        # hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return attention_probs



class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.query_linear = nn.Linear(input_size, hidden_size)
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)
        
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        attention_probs_dropout_prob = 0.2

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        
    def forward(self, inputs):
        batch_size, seq_length, _ = inputs.size()
        inputs = inputs.float()
        # Linear transformations for query, key, value
        query = self.query_linear(inputs)
        key = self.key_linear(inputs)
        value = self.value_linear(inputs)
        
        # Reshape query, key, value to enable multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, seq_length, hidden_size//num_heads)
        key = key.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)      # (batch_size, num_heads, seq_length, hidden_size//num_heads)
        value = value.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, seq_length, hidden_size//num_heads)
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size // self.num_heads, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)

        attention_weights = self.attn_dropout(attention_weights)
        
        # Apply attention weights to value vectors
        attended_values = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_length, hidden_size//num_heads)
        
        # Concatenate and reshape the attended values
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # (batch_size, seq_length, hidden_size)
        
        # Linear transformation of concatenated values
        output = self.output_linear(attended_values)
        
        return  attention_weights
    



class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算查询、键和值
        x = x.float()
        query = self.query(x).view(batch_size, -1, self.hidden_dim)
        key = self.key(x).view(batch_size, -1, self.hidden_dim)
        value = self.value(x).view(batch_size, -1, self.hidden_dim)
        
        # 计算注意力分数
        scores = torch.bmm(query, key.transpose(1, 2))
        # attention_weights = self.softmax(scores).squeeze(dim=0)
        attention_weights = (scores).squeeze(dim=0)
        
        # 加权求和
        return attention_weights
    
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
        
        x = x.view(-1,32,2,2)
        
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) # no relu

        out = out.view(out.size(0),-1) # bs*1

        return out

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        return attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, dropout=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk

        attn = self.attention(q, k)

        return attn
    

class ResNet12Block(nn.Module):
    """
    ResNet Block
    """
    def __init__(self, inplanes, planes):
        super(ResNet12Block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out




class ResNet12(nn.Module):
    """
    ResNet12 Backbone
    """
    def __init__(self, emb_size=1600, block=ResNet12Block, cifar_flag=False):
        super(ResNet12, self).__init__()
        cfg = [64, 128, 256, 512]
        # layers = [1, 1, 1, 1]
        iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU()
        self.emb_size = emb_size
        self.layer1 = self._make_layer(block, cfg[0], cfg[0])
        self.layer2 = self._make_layer(block, cfg[0], cfg[1])
        self.layer3 = self._make_layer(block, cfg[1], cfg[2])
        self.layer4 = self._make_layer(block, cfg[2], cfg[3])
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AvgPool2d(6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        self.layer_second = nn.Sequential(nn.Linear(in_features=layer_second_in_feat,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes):
        layers = []
        layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 3 -> 64
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 64 -> 64
        x = self.layer1(x)
        # 64 -> 128
        x = self.layer2(x)
        # 128 -> 256
        inter = self.layer3(x)
        # 256 -> 512
        # print(inter.shape)
        x = self.layer4(inter)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # 512 -> 128
        x = self.layer_last(x)
        # print(x.shape)
        inter = self.maxpool(inter)
        # 256 * 5 * 5
        inter = inter.view(inter.size(0), -1)
        # 256 * 5 * 5 -> 128
        inter = self.layer_second(inter)
        out = []
        # out.append(x)
        out.append(inter)
        # no FC here
        return inter

