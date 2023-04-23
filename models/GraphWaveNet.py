'''
GraphWaveNet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import pandas as pd
import scipy.sparse.linalg as linalg
import pickle

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_adj(pkl_filename, adjtype):
    try:
        # METRLA and PEMSBAY
        _, _, adj_mx = load_pickle(pkl_filename)
    except ValueError:
        # PEMS3478
        adj_mx = load_pickle(pkl_filename)
        
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
        self.meta_att = None

    def set_meta_att(self, A):
        self.meta_att = A

    def forward(self,x, A):
        # print('x', x.shape)
        if len(A.shape)==2:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        if self.meta_att is not None:
            x = torch.einsum('bin,bcnk->bcik',(self.meta_att,x))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        self.meta_adj = None
        self.meta_att = None

    def set_meta_adj(self, G):
        self.meta_adj = G

    def set_meta_att(self, A):
        self.nconv.set_meta_att(A)

    def forward(self,x,support):
        out = [x]
        if self.meta_adj is not None:
            x1 = self.nconv(x,self.meta_adj)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,self.meta_adj)
                out.append(x2)
                x1 = x2

        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        
        h = torch.cat(out,dim=1)
        # print('h', h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(nn.Module):
    def __init__(self, 
                 device, 
                 num_nodes, 
                 dropout=0.3, 
                 adj_path=None,
                 adj_type=None, 
                 gcn_bool=True, 
                 addaptadj=True, 
                 aptinit=None, 
                 in_dim=1,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                 add_meta_adj=False,
                 add_meta_att=False,
                 node_emb_file=None,
                 tod_embedding_dim=24,
                 dow_embedding_dim=7,
                 node_embedding_dim=64,
                 learner_hidden_dim=128,
                 z_dim=32,
                 in_steps=12
                 ):
        super(GWNET, self).__init__()
        if adj_path:
            adj_mx = load_adj(adj_path, adj_type)
            supports = [torch.tensor(i).to(device) for i in adj_mx]
        else:
            supports = None
        
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.add_meta_adj = add_meta_adj
        self.add_meta_att = add_meta_att
        self.use_meta = self.add_meta_adj or self.add_meta_att
        self.st_embedding_dim = (
            tod_embedding_dim + dow_embedding_dim + node_embedding_dim
        )
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

    
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))
                
                # ! 以下三个 conv1d 因为 torch1.11 以上版本的不兼容 改成了 conv2d
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

        if self.use_meta:
            self.node_embedding = torch.FloatTensor(np.load(node_emb_file)["data"]).to(device)

            self.tod_onehots = torch.eye(24, device=device)
            self.dow_onehots = torch.eye(7, device=device)  
            
            if self.z_dim > 0:
                self.mu = nn.Parameter(torch.randn(num_nodes, z_dim), requires_grad=True)
                self.logvar = nn.Parameter(
                    torch.randn(num_nodes, z_dim), requires_grad=True
                )

                self.mu_estimator = nn.Sequential(
                    nn.Linear(in_steps, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, z_dim),
                )

                self.logvar_estimator = nn.Sequential(
                    nn.Linear(in_steps, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, z_dim),
                )    

            if self.add_meta_adj:
                self.adj_learner = nn.Sequential(
                    nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, 30),
                )    
            if self.add_meta_att:
                self.att_learner = nn.Sequential(
                    nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, 30),
                )                


    def forward(self, input): # ! (B, C, N, T)
        batch_size = input.shape[0]

        if self.use_meta:
            tod = input[..., 1]  # (B, T_in, N)
            dow = input[..., 2]  # (B, T_in, N)
            input = input[..., :1]  # (B, T_in, N, 1)
                    # use the last time step to represent the temporal location of the time seires
            tod_embedding = self.tod_onehots[(tod[:, -1, :] * 24).long()]  # (B, N, 24)
            dow_embedding = self.dow_onehots[dow[:, -1, :].long()]  # (B, N, 7)
            node_embedding = self.node_embedding.expand(
                batch_size, *self.node_embedding.shape
            )  # (B, N, node_emb_dim)
            
            meta_input = torch.concat(
                [node_embedding, tod_embedding, dow_embedding], dim=-1
            )  # (B, N, st_emb_dim)

            if self.z_dim > 0:
                z_input = input.squeeze(dim=-1).transpose(1, 2)

                mu = self.mu_estimator(z_input)  # (B, N, z_dim)
                logvar = self.logvar_estimator(z_input)  # (B, N, z_dim)

                z_data = self.reparameterize(mu, logvar)  # temporal z (B, N, z_dim)
                z_data = z_data + self.reparameterize(
                    self.mu, self.logvar
                )  # temporal z + spatial z
                
                meta_input = torch.concat(
                    [meta_input, z_data], dim=-1
                )  # (B, N, st_emb_dim+z_dim)

            if self.add_meta_adj:
                adj_embeddings = self.adj_learner(meta_input)
                meta_adp = F.softmax(F.relu(torch.einsum('bih,bhj->bij', [adj_embeddings, adj_embeddings.transpose(1, 2)])), dim=-1)
                for i in range(self.blocks * self.layers):
                    self.gconv[i].set_meta_adj(meta_adp)

            if self.add_meta_att:
                att_embeddings = self.att_learner(meta_input)
                meta_att = F.softmax(F.relu(torch.einsum('bih,bhj->bij', [att_embeddings, att_embeddings.transpose(1, 2)])), dim=-1)
                for i in range(self.blocks * self.layers):
                    self.gconv[i].set_meta_att(meta_att)               


        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0  

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None and not self.add_meta_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj and not self.use_meta :
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) # 这里不用 permute 回去 维度顺序是正好对的
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
  