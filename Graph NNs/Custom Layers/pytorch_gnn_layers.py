import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGCNLayer(nn.Module):

    """
        Custom GCN Layer.
        Paper: https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.lin = nn.Linear(in_dims, out_dims, bias=False)
        
    def forward(self, adj, inputs):
        #adj = NxN adjacency Matrix
        #inputs = graph initial embeddings

        a_tilde = adj + torch.eye(adj.shape[0]) #add self loops
        
        #Degree Matrix
        deg = torch.diag(adj.sum(1))
        
        inv_sqr_deg = torch.linalg.inv(torch.pow(deg,0.5))
        a_hat = inv_sqr_deg @ a_tilde @ inv_sqr_deg
        
        return self.lin(a_hat@inputs)


class CustomGATLayer(nn.Module):

    """
        Simple GAT Layer.
        Reference: https://github.com/Diego999/pyGAT/blob/master/layers.py
        Paper: https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_dims, out_dims, alpha=0.2, dropout=0.1):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.dropout = dropout
        
        # Weight init
        self.W = nn.Parameter(torch.empty(size=(in_dims, out_dims)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_dims,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        
    def forward(self, adj, inputs):
        #adj = NxN adjacency Matrix
        #inputs = graph initial embeddings
        
        adj = adj + torch.eye(adj.shape[0]) #add self loops
        Wh = inputs @ self.W
        
        N = Wh.shape[0]
        
        # Unnormalized Attention Coefficients
        Wh1 = Wh.repeat_interleave(N, dim=0) #N*N, out_dim
        Wh2 = Wh.repeat(N, 1)
        
        #Build All combination matrix, e1||e2, ... e1||en .... en||en, then mask
        combination_matrix = torch.cat([Wh1, Wh2], dim=1) #N*N, 2*out
        combination_matrix = combination_matrix.reshape((N,N,2*self.out_dims))
        e = combination_matrix @ self.a #N,N,1
        e = e.squeeze(2) #n,n
        e = self.leaky_relu(e)
        
        #Masked Attention
        zero_mat = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_mat)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        out = attention @ Wh
        
        return out                


class CustomGATLayerEff(nn.Module):

    """
        Memory Efficient GAT Layer implementation.. performance to be evaluated
        Reference: https://github.com/Diego999/pyGAT/blob/master/layers.py
        Paper: https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_dims, out_dims, alpha=0.2, dropout=0.1):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.dropout = dropout
        
        # Weight init
        self.W = nn.Parameter(torch.empty(size=(in_dims, out_dims)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_dims,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        
    def forward(self, adj, inputs):
        #adj = NxN adjacency Matrix
        #inputs = graph initial embeddings
        
        adj = adj + torch.eye(adj.shape[0]) #add self loops
        
        Wh = inputs @ self.W
        
        # Unnormalized Attention Coefficients
        Wh1 = Wh @ self.a[:self.out_dims, :] #N, 1
        Wh2 = Wh @ self.a[self.out_dims:, :] #N, 1
        
        #Broadcast sum
        e = Wh1 + Wh2.T #Unnormalized Att Coefficients, NxN
        e = self.leaky_relu(e)
        
        #Masked Attention
        zero_mat = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_mat)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        out = attention @ Wh
        
        return out                