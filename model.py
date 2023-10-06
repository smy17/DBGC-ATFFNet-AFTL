import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data



# 生成邻接矩阵
class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio=128):
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias=False),
                                nn.ELU(inplace=False),
                                nn.Linear(inc // reduction_ratio, inc, bias=False),
                                nn.Tanh(),
                                nn.ReLU(inplace=False))

    def forward(self, x):
        y = self.fc(x)
        return y


class resGCN(nn.Module):
    def __init__(self, inc, outc,band_num):
        super(resGCN, self).__init__()
        self.GConv1 = nn.Conv2d(in_channels=inc,
                                out_channels=outc,
                                kernel_size=(1, 3),
                                stride=(1, 1),
                                padding=(0, 0),
                                groups=band_num,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.GConv2 = nn.Conv2d(in_channels=outc,
                                out_channels=outc,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 1),
                                groups=band_num,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_p, L):
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        y = torch.einsum('bijk,kp->bijp', (x, L))
        y = self.ELU(torch.add(y, x_p))
        return y



class HGCN(nn.Module):
    def __init__(self, dim, chan_num,band_num, reduction_ratio, si):
        super(HGCN, self).__init__()
        self.chan_num = chan_num
        self.dim = dim
        self.resGCN = resGCN(inc=dim * band_num,
                              outc=dim * band_num,band_num=band_num)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x,A_ds):
        x = x.permute(0,2,1).unsqueeze(2)
        L = torch.einsum('ik,kp->ip', (A_ds, torch.diag(torch.reciprocal(sum(A_ds)))))
        G = self.resGCN(x, x, L).contiguous()
        return G.squeeze(2).transpose(2,1)



    
# Transformer 部分
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_head = 5

    def forward(self, Q, K, V,A_ds):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        #scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        chan_weighting = A_ds.repeat(self.n_head,1,1)
        attn = attn*chan_weighting
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.adapter = Adapter(d_model)
        
    def forward(self, input_Q, input_K, input_V,A_ds):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention(self.d_k)(Q, K, V,A_ds)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        output = self.adapter(output)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.adapter = Adapter(d_model)
        
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        output = self.adapter(output)
        return nn.LayerNorm(self.d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads,d_model,d_k,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, enc_inputs,A_ds):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,A_ds) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self,n_layers,n_heads,d_model,d_k,d_v,d_ff):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_heads,d_model,d_k,d_v,d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs,A_ds):
        enc_outputs = enc_inputs
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs = layer(enc_outputs,A_ds)
        return enc_outputs


# 增添Adapter 模块
class Adapter(nn.Module):
    def __init__(self,d1):
        super(Adapter,self).__init__()
        self.ada_linear1 = nn.Linear(d1,8)
        self.ELU = nn.ELU(inplace=False)
        self.ada_linear2 = nn.Linear(8,d1)
        
    def forward(self,x):
        x1 = self.ELU(self.ada_linear1(x))
        x2 = self.ada_linear2(x1)
        return x+x2
 
class DAGCN(nn.Module):
    def __init__(self,dataset):
        super(DAGCN,self).__init__()
        options = {'seed':[62,3,5],'seed iv':[62,4,5],'deap':[32,4,4]}
        self.chan_num = options[dataset][0]
        self.class_num = options[dataset][1]
        self.band_num = options[dataset][2]
        self.gcn1 = HGCN(dim = 1, chan_num = self.chan_num,band_num=self.band_num, reduction_ratio = 128, si = 256)
        self.gcn2 = HGCN(dim = 1, chan_num = self.chan_num,band_num=self.band_num, reduction_ratio = 128, si = 256)
        self.encoder = Encoder(n_layers=2,n_heads=5,d_model=self.band_num*2,d_k=8,d_v=8,d_ff=10)
        self.encoder1 = Encoder(n_layers=2,n_heads=5,d_model=self.band_num,d_k=8,d_v=8,d_ff=10)
        self.encoder2 = Encoder(n_layers=2,n_heads=5,d_model=self.band_num,d_k=8,d_v=8,d_ff=10)
        self.linear = nn.Linear(self.chan_num*self.band_num*2,64)
        self.A = torch.rand((1, self.chan_num * self.chan_num), dtype=torch.float32, requires_grad=False).cuda()
        self.GATENet = GATENet(self.chan_num * self.chan_num, reduction_ratio=128)
        self.linear2 = nn.Linear(64,self.class_num)

        self.fc = nn.Linear(self.chan_num*self.band_num*2,64)

    def forward(self,x):
        #[n, 32, 8]
        A_ds = self.GATENet(self.A)
        A_ds = A_ds.reshape(self.chan_num, self.chan_num)
        de = x[:,:,:self.band_num]
        psd = x[:,:,self.band_num:]
        feat1 = self.encoder1(self.gcn1(de,A_ds),A_ds)
        feat2 = self.encoder2(self.gcn2(psd,A_ds),A_ds)
        
        #[n, 32, 8]
        feat0 = torch.cat([feat1,feat2],dim=2)
        feat = self.encoder(feat0,A_ds)
        feat = feat.reshape(-1,self.chan_num*self.band_num*2)
        feat = self.linear(feat)
        out = self.linear2(feat)        
        
        tsne = feat.reshape(x.shape[0],-1) #feat.view(x.shape[0],-1)
        return out, tsne