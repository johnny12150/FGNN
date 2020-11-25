# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv
from torch_geometric.utils import subgraph
import weighted_gat


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size, heads=1):
        super(Embedding2Score, self).__init__()
        self.hidden_size = heads * hidden_size
        self.heads = heads
        if self.heads > 1:
            self.multi_int = True
        else:
            self.multi_int = False
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        q1 = self.W_1(torch.cat(v_n_repeat, dim=0))  # ht
        q2 = self.W_2(session_embedding)  # entire session
        alpha = self.q(torch.sigmoid(q1 + q2))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)  # global session emb
        
        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)  # ht
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))  # concat s_g & s_l

        if self.multi_int:
            h = int(self.hidden_size/ self.heads)
            s_h = s_h.view(-1, h)
        # Eq(8)
        z_i_hat = torch.mm(s_h, all_item_embedding.weight.transpose(1, 0))
        if self.multi_int:
            z_i_hat = z_i_hat.view(-1, self.heads, all_item_embedding.weight.shape[0])
            z_i_hat = torch.max(z_i_hat, 1).values
        
        return z_i_hat


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.heads = 16
        self.interest = 3
        gats = []
        # todo 看一下 GATCONv的 att (self.gat1.att)
        self.gat1 = GATConv(self.hidden_size, self.hidden_size, heads=self.heads, negative_slope=0.2)
        self.gat2 = GATConv(self.heads * self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2)
        for i in range(self.interest):
            gats.append(GATConv(self.hidden_size, self.hidden_size, heads=self.heads, negative_slope=0.2))
            gats.append(GATConv(self.heads * self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2))
        self.gats = nn.ModuleList(gats)
        self.sage1 = SAGEConv(self.hidden_size, self.hidden_size)
        self.sage2 = SAGEConv(self.hidden_size, self.hidden_size)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=2)
        self.to_interest = nn.Linear(self.hidden_size, self.interest * self.hidden_size)
        ggnns = [GatedGraphConv(self.hidden_size, num_layers=1) for i in range(self.interest)]
        self.ggnns = nn.ModuleList(ggnns)
        self.w_interest = nn.Linear(self.hidden_size, self.interest, bias=False)

        self.wGAT1 = weighted_gat.WeightedGATConv(self.hidden_size, self.hidden_size, heads=self.heads, negative_slope=0.2)
        self.wGAT2 = weighted_gat.WeightedGATConv(self.heads * self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2)

        self.e2s = Embedding2Score(self.hidden_size, self.interest)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x - 1, data.edge_index, data.batch, data.edge_attr
        # print(torch.sum(data.sequence_len), data.sequence.shape[0])  # sequence是batch下全部concat
        # print(x.squeeze(dim=-1).unique().shape, x.shape)  # x是batch下總共有多少edge

        embedding = self.embedding(x).squeeze()

        # wGAT
        # hidden = F.relu(self.wGAT1(embedding, edge_index, edge_attr))
        # hidden = self.wGAT2(hidden, edge_index, edge_attr)

        # GGNN
        # hidden = self.gated(embedding, edge_index)

        # 開大矩陣直接轉interest emb
        # embedding = self.to_interest(embedding).view(-1, self.interest, self.hidden_size)

        alpha_item_interest = self.w_interest(embedding)
        g_item_interest = F.softmax(alpha_item_interest / 0.1, dim=-1)
        embedding = torch.einsum('ik,il->ikl', g_item_interest, embedding)

        # perform drop node, todo add shortcut graph
        # mask node
        interest_mask = (g_item_interest > 1/ self.interest).T
        # x_copies = x.repeat((self.interest, 1)).view(self.interest, -1)
        # mask select
        # x_masked = torch.masked_select(x_copies, interest_mask)  # 和input shape會不同
        # x_masked = [x[interest_mask[i]].flatten() for i in range(self.interest)]  # get original node id
        x_masked = [interest_mask[i].flatten() for i in range(self.interest)]
        # sub graph
        # todo add shortcut
        edge_indexes = [subgraph(x_masked[i], edge_index)[0] for i in range(self.interest)]  # drop node and edge by mask

        hiddens = []
        for i in range(self.interest):
            # hidden = self.ggnns[i](embedding[:, i], edge_index)
            # hidden = F.relu(self.gats[i*2](embedding[:, i], edge_index))
            # hidden = self.gats[i*2+1](hidden, edge_index)
            hidden = F.relu(self.gats[i*2](embedding[:, i], edge_indexes[i]))
            hidden = self.gats[i*2+1](hidden, edge_indexes[i])
            hiddens.append(hidden)
        hidden = torch.cat(hiddens, 1)  # concat all interest emb

        #  產 interest session embedding
        # hidden = self.gat1(embedding, edge_index)

        # GAT
        # hidden = F.relu(self.gat1(embedding, edge_index))
        # hidden = self.gat2(hidden, edge_index)

        # SAGE
        # hidden1 = F.relu(self.sage1(embedding, edge_index))
        # hidden2 = F.relu(self.sage2(hidden1, edge_index))

        return self.e2s(hidden, self.embedding, batch)
