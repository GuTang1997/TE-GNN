
#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
@author: Gu Tang
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class Time_GNN(Module):
    def __init__(self,hidden_size,step=2):
        super(Time_GNN,self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.linear_graph = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.linear_trans_h = nn.Linear(self.hidden_size,1,bias=False)
        self.linear_gate = nn.Linear(2 * self.hidden_size,1,bias=False)
        self.linear_or = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.w_g = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.Q = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.K = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
    def Time_gnn_cell(self,graph,hidden,masks):
        eye_mat = torch.eye(graph.shape[1]).unsqueeze(0).cuda()
        t_graph = (graph + eye_mat)*masks
        time_gnn_hidden = torch.matmul(t_graph,hidden)
        out =  time_gnn_hidden 

        return out
    def forward(self,graph,hidden,masks):
        or_hidden = hidden
#        mask_h = (masks - 1.)*9e19
#        mask_last = masks.squeeze(-1).long()
        for i in range(self.step):
            hidden = self.Time_gnn_cell(graph,hidden,masks)
        gate = torch.sigmoid(self.linear_gate(torch.cat([hidden,or_hidden],dim=-1)))
        out =gate * hidden  +  (1-gate) * or_hidden
        return out
    
class Time_awar_attention(Module):
    def __init__(self,hidden_size,bi):
        super(Time_awar_attention,self).__init__()  
        self.hidden_size = hidden_size
        self.bi = bi
        self.pos_embedding = nn.Embedding(200,self.hidden_size)
        self.filed_pos = nn.Embedding(50,self.hidden_size)
        self.pos_linear = nn.Linear(self.hidden_size *2, self.hidden_size)
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))   
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.w_3 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.w_pos = nn.Parameter(torch.Tensor(self.hidden_size * 2,self.hidden_size))
        self.Gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=self.bi, batch_first=True)
        self.line_gate = nn.Linear(self.hidden_size *2, 1,bias=False)
    def gru_forward(self,gru, input, lengths, state=None, batch_first=True):
        gru.flatten_parameters()
        input_lengths, perm = torch.sort(lengths, descending=True)
        input = input[perm]
        if state is not None:
            state = state[perm].transpose(0, 1).contiguous()
        total_length=input.size(1)
        if not batch_first:
            input = input.transpose(0, 1)  # B x L x N -> L x B x N
        packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)
        outputs, state = gru(packed, state)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)
        _, perm = torch.sort(perm, descending=False)
        if not batch_first:
            outputs = outputs.transpose(0, 1)
        outputs=outputs[perm]
        state = state.transpose(0, 1)[perm]
        return outputs, state
    def forward(self,hidden,time_weight,mask,short,R=False,decay=10):
        seq_lens = mask.float().sum(dim=-1).long()
        t = time_weight

        masks = mask.float().unsqueeze(-1)
        res = t.squeeze(-1).long() * mask
#        print(res[0])
        if R== True:
            short = hidden[torch.arange(mask.shape[0]).long(),0]
        else:
            short = hidden[torch.arange(mask.shape[0]).long(),torch.sum(mask, 1).long() - 1]
        lens = hidden.shape[1]
        batch_size = hidden.shape[0]

        filed_pos_emb = self.filed_pos(res)

        filed_pos_emb,state = self.gru_forward(self.Gru,filed_pos_emb,seq_lens,batch_first=True)
        if self.bi == True: 
            
            filed_pos_emb = filed_pos_emb.view(batch_size,lens,2,-1)
            back_filed_pos_emb = filed_pos_emb[:,:,-1]
            for_filed_pos_emb = filed_pos_emb[:,:,-2]
            gnn_gate = self.line_gate(torch.cat([back_filed_pos_emb,for_filed_pos_emb],dim=-1))
            filed_pos_emb = back_filed_pos_emb * gnn_gate + (1.-gnn_gate) * for_filed_pos_emb

        en_pos = torch.matmul(torch.cat([filed_pos_emb,hidden],dim=-1),self.w_pos)
        en_pos_t = torch.tanh(en_pos)
        pos_score = torch.matmul(en_pos_t,self.w_2) * masks
        out = torch.sum(pos_score * hidden, 1)
        return torch.dropout(out,p=0.3,train=self.training)
    
class Time_Model(Module):
    def __init__(self,opt,n_items,mask_value = None):
        super(Time_Model,self).__init__()
        self.mask_value = mask_value
        self.hidden_size = opt.hiddenSize
        self.l_size = 256
        self.n_items = n_items
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_items,self.hidden_size)
        self.linear_K = nn.Linear(self.hidden_size,self.hidden_size * 2,bias=False)
        self.linear_Q = nn.Linear(self.hidden_size,self.hidden_size * 2,bias=False)
        self.linear_K_1 = nn.Linear(self.hidden_size,self.hidden_size * 2,bias=False)
        self.linear_Q_1 = nn.Linear(self.hidden_size,self.hidden_size * 2,bias=False)
        self.linear_cat = nn.Linear(self.hidden_size,1,bias=False)
        self.linear_cat_1 = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.enc = nn.GRU(self.hidden_size , self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)

        self.enc_1 = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.t_gnn = Time_GNN(self.hidden_size)
        self.t_att = Time_awar_attention(self.hidden_size,bi=True)
        self.t_att_GNN = Time_awar_attention(self.hidden_size,bi=True)
        
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def Multi_head_mask_avg(self,scores,mask):
        scores = scores.cuda()
        mask = mask.cuda()
        mask_inputs = scores * mask.view(mask.shape[0],1,1,-1).float()
        eye_mat = torch.eye(scores.shape[-1]).unsqueeze(0).cuda()
        mask_mat = torch.where(eye_mat>0.,torch.tensor(0.).cuda(),torch.tensor(1.).cuda(),)
        mask_result = mask_inputs * mask_mat
        mask_result_sum = mask_result.sum(dim=-1) / (mask.sum(dim=-1)).view(mask.shape[0],1,1).float()
        mask_result_sum_mask = mask_result_sum + (mask.unsqueeze(1).float() - 1.) *1e9
        res = torch.softmax(mask_result_sum_mask,dim=-1)
        return res

    def Multi_Head(self,x,mask,num_head=4):
        d_k = x.size(-1)
        n_batch = x.size(0)
        Q =self.linear_Q(x)
        K = self.linear_K(x)
        Q,K = [h.view(n_batch,-1,num_head,d_k).transpose(1,2) for h in (Q,K)]
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(d_k)

        att_score = self.Multi_head_mask_avg(scores,mask)
        res = x * att_score.mean(1).unsqueeze(-1)
        return res
    def gru_forward(self,gru, input, lengths, state=None, batch_first=True):
        gru.flatten_parameters()
        input_lengths, perm = torch.sort(lengths, descending=True)

        input = input[perm]
        if state is not None:
            state = state[perm].transpose(0, 1).contiguous()

        total_length=input.size(1)
        if not batch_first:
            input = input.transpose(0, 1)  # B x L x N -> L x B x N
        packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)

        outputs, state = gru(packed, state)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

        _, perm = torch.sort(perm, descending=False)
        if not batch_first:
            outputs = outputs.transpose(0, 1)
        outputs=outputs[perm]
        state = state.transpose(0, 1)[perm]

        return outputs, state
    def forward(self,inputs,gnn_items,A,t_graph,alias_inputs,mask):
        
        masks = mask.float().unsqueeze(-1)
        lengths = mask.float().sum(dim=-1).long()
        item_embed = self.embedding(inputs[0])
        gnn_embed = self.embedding(gnn_items)
        short = item_embed[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1).long() - 1] 
        time_weight = inputs[1]
        
        gnn_time_weight = inputs[2]
        gnn_time_ex = gnn_time_weight.view((time_weight.shape[0],time_weight.shape[1],1))

        ''' time gnn '''       
        t_gnn_embed = self.t_gnn(t_graph,gnn_embed,masks)

        GNN_embed = torch.dropout(t_gnn_embed,p=0.3,train=self.training)
        select = self.t_att_GNN(GNN_embed,gnn_time_ex,mask,short,R=True,decay=16)
        '''merg gnn''' 
        item_embed_gnn = torch.dropout(t_gnn_embed,p=0.5,train=self.training) #Diginetica :0. ,others:0.5
        out, state = self.gru_forward(self.enc, item_embed_gnn, lengths, batch_first=True)
        state = state.view(-1,self.hidden_size)
        z_h_long = select
        gate = torch.sigmoid(self.linear_cat((z_h_long+state)))
        z_h = z_h_long * gate + (1. - gate)*state 

        item_embed_weights = self.embedding.weight[1:]
        output = torch.matmul(z_h,item_embed_weights.transpose(1, 0))
        return output

    def Mask_avg(self,scores,mask):
        scores = scores.cuda()
        mask = mask.cuda()
        mask_inputs = scores * mask.view(mask.shape[0],1,-1).float()
        eye_mat = torch.eye(scores.shape[1]).cuda()
        mask_mat = torch.where(eye_mat>0.,torch.tensor(0.).cuda(),torch.tensor(1.).cuda(),)
        mask_result = mask_inputs * mask_mat
        mask_result_sum = mask_result.sum(dim=-1) / (mask.sum(dim=-1)).view(mask.shape[0],1).float()
        mask_result_sum_mask = mask_result_sum + (mask.float() - 1.) *1e9
        return mask_result_sum_mask
        
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A,t_graph,inputs, mask, targets = data.get_slice(i)
    items = inputs[0]
    gnn_items = inputs[1]
    time_weight = inputs[2]
    gnn_time_weight = inputs[3]
    items = trans_to_cuda(torch.Tensor(items).long())
    gnn_items = trans_to_cuda(torch.Tensor(gnn_items).long())
    time_weight = trans_to_cuda(torch.Tensor(time_weight).float())
    gnn_time_weight = trans_to_cuda(torch.Tensor(gnn_time_weight).float())
    A = trans_to_cuda(torch.Tensor(A).float())
    t_graph = trans_to_cuda(torch.Tensor(t_graph).float())
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    
    hidden = model([items,time_weight,gnn_time_weight],gnn_items,A,t_graph,alias_inputs,mask)
    return targets, hidden


def train_test(model, train_data, test_data):
    
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data) 
        targets = trans_to_cuda(torch.Tensor(targets).long())

        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
