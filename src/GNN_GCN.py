import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init, Linear
import dgl
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv
from model_configurations import set_block, set_function
from base_classes import BaseGNN
from functools import partial
#code adapted from https://github.com/kdd-submitter/on_local_aggregation/blob/main/models.py - converted to PyG

class PyG_GCN(nn.Module):
    def __init__(self, in_feat, out_feat, **layer_kwargs):
        super(PyG_GCN, self).__init__()
        self.GCNConv = GCNConv(in_feat, out_feat)

    def forward(self, graph, features):
        edge_index = torch.cat([graph.edges()[0].unsqueeze(0), graph.edges()[1].unsqueeze(0)], dim=0)
        return self.GCNConv(features, edge_index)


class GraphSequential(nn.Module):
    def __init__(self, layer_stack, opt):
        super().__init__()
        self.layer_stack = nn.ModuleList(layer_stack)
        self.gcn_layer_types = [GraphConv, PyG_GCN, GCNConv]
        self.opt = opt

    def forward(self, graph, features):
        for layer in self.layer_stack:
            if any([isinstance(layer, gcn_type) for gcn_type in self.gcn_layer_types]):
                if self.opt['gcn_symm']:
                    # encoder conv
                    if layer._in_feats != layer._out_feats:
                        features = layer(graph, features)
                    # symmetric gcn
                    elif self.opt['function'] == 'gcn_dgl':
                        symm_weight = (layer.symm_weight + layer.symm_weight.T) / 2
                        features = layer(graph, features, weight=symm_weight)
                    # symmetric res gcn
                    elif self.opt['function'] == 'gcn_res_dgl':
                        symm_weight = (layer.symm_weight + layer.symm_weight.T) / 2
                        features = features + self.opt['step_size'] * layer(graph, features, weight=symm_weight)
                else:
                    if self.opt['function'] == 'gcn_dgl':
                        features = layer(graph, features)
                    elif self.opt['function'] == 'gcn_res_dgl':
                        if layer._in_feats != layer._out_feats:
                            features = layer(graph, features)
                        else:
                            features = features + self.opt['step_size'] * layer(graph, features)
                    elif self.opt['function'] == 'gcn_pyg':
                        features = layer(features, graph) ##here "graph" is edge index
            else:
                features = layer(features)
        return features


class GCNs(BaseGNN):
    def __init__(self, opt, dataset, device, feat_repr_dims, dropout=0.0):
        ###required for code homogeniety like counting NFEs
        super(GCNs, self).__init__(opt, dataset, device)
        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, opt, dataset.data, device, t=time_tensor).to(device)

        if opt['gcn_symm']:
            assert opt['function'] in ['gcn_dgl','gcn_res_dgl'], 'gcn_symm only implenent for dgl conv'

        self.opt = opt
        self.edge_index = dataset.data.edge_index
        dims = list(zip(feat_repr_dims[:-1], feat_repr_dims[1:]))
        self.nonlinearity = nn.ReLU

        ### choice of PyG and DGL GCNConv
        if self.opt['function'] == 'gcn_dgl':
            gcn_layer_type = GraphConv
        elif self.opt['function'] == 'gcn_res_dgl':
            gcn_layer_type = GraphConv
        elif self.opt['function'] == 'gcn_pyg':
            gcn_layer_type = GCNConv

        gcn_kwargs = {}
        self.gcn_stack = self._make_stack(dims, dropout, gcn_layer_type, gcn_kwargs)


    def _make_stack(self, dims, dropout, layer_type, layer_kwargs):
        stack = []

        if self.opt['gcn_enc_dec']:
            self.m1 = nn.Linear(dims[0][0], dims[0][1])
            stack_dims = dims[1:-1]
        else:
            stack_dims = dims

        #initialise the fixed shared layer if required
        if self.opt['gcn_fixed']:
            if self.opt['gcn_enc_dec']:
                if self.opt['gcn_symm']:
                    #init layer without weights
                    GCN_fixedW = layer_type(stack_dims[0][0], stack_dims[0][1], weight=False, bias=self.opt['gcn_bias'], **layer_kwargs)
                    #insert parameter
                    GCN_fixedW.symm_weight = nn.Parameter(torch.Tensor(stack_dims[0][0], stack_dims[0][1]))
                    init.xavier_uniform_(GCN_fixedW.symm_weight)
                else:
                    GCN_fixedW = layer_type(stack_dims[0][0], stack_dims[0][1], bias=self.opt['gcn_bias'],  **layer_kwargs)
            else:
                if self.opt['gcn_symm']:
                    GCN_fixedW = layer_type(stack_dims[1][0], stack_dims[1][1], weight=False, bias=self.opt['gcn_bias'], **layer_kwargs)
                    GCN_fixedW.symm_weight = nn.Parameter(torch.Tensor(stack_dims[1][0], stack_dims[1][1]))
                    init.xavier_uniform_(GCN_fixedW.symm_weight)
                else:
                    GCN_fixedW = layer_type(stack_dims[1][0], stack_dims[1][1], bias=self.opt['gcn_bias'], **layer_kwargs)


        for indx, (in_feat, out_feat) in enumerate(stack_dims):
            if self.opt['gcn_mid_dropout']:
                stack.append(nn.Dropout(dropout))

            if in_feat != out_feat: #overide to ignore fixed W or residual blocks if using a convolutional encoder/decoder
                #note can't have a symmetric W here
                stack.append(layer_type(in_feat, out_feat, bias=self.opt['gcn_bias'], **layer_kwargs))
            elif self.opt['gcn_fixed']:
                stack.append(GCN_fixedW)
            else:
                if self.opt['gcn_symm']:
                    layerConv = layer_type(in_feat, out_feat, weight=False, bias=self.opt['gcn_bias'], **layer_kwargs)
                    layerConv.symm_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
                    init.xavier_uniform_(layerConv.symm_weight)
                else:
                    layerConv = layer_type(in_feat, out_feat, bias=self.opt['gcn_bias'], **layer_kwargs)
                stack.append(layerConv)

            if indx < len(stack_dims) - 1:
                if self.opt['gcn_non_lin']:
                    stack.append(self.nonlinearity())

        if self.opt['gcn_enc_dec']:
            self.m2 = nn.Linear(stack_dims[0][0], stack_dims[0][1])

        return GraphSequential(stack, self.opt)

    def encoder(self, features, pos_encoding=None):
        if self.opt['gcn_enc_dec']:
            features = F.dropout(features, self.opt['input_dropout'], training=self.training)
            features = self.m1(features)
        return features

    def forward_XN(self, features):
        if self.opt['function'] in ['gcn_dgl', 'gcn_res_dgl']:
            graph = dgl.graph((self.edge_index[0], self.edge_index[1])).to(self.device)
        elif self.opt['function'] == 'gcn_pyg':
            graph = self.edge_index

        features = self.encoder(features, pos_encoding=None)
        features = self.gcn_stack(graph, features)
        return features

    def forward(self, features, pos_encoding=None): #run_GNN.py

        features = self.forward_XN(features)

        if self.opt['gcn_enc_dec']:
            features = self.m2(features)

        return features

class MLP(BaseGNN):
  def __init__(self, opt, dataset, device=None):
    ###required for code homogeniety like counting NFEs
    super(MLP, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, opt, dataset.data, device, t=time_tensor).to(device)

    self.opt = opt
    self.m1 = Linear(dataset.data.x.shape[1], opt['hidden_dim'])
    self.m2 = Linear(opt['hidden_dim'], dataset.num_classes)

    self.epoch = 0
    self.wandb_step = 0

  def forward(self, x, pos_encoding): #todo pos_encoding
    x = F.dropout(x, self.opt['dropout'], training=self.training)
    x = F.dropout(self.m1(torch.tanh(x)), self.opt['dropout'], training=self.training)
    x = F.dropout(self.m2(torch.tanh(x)), self.opt['dropout'], training=self.training)

    return x