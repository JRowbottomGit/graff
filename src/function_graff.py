import math
import os
import shutil
import torch
from torch import nn
from torch.nn.init import uniform, xavier_uniform_
import torch.nn.functional as F
import torch_sparse
from torch_geometric.utils import degree, softmax, homophily
from torch_geometric.nn.inits import glorot, zeros, ones, constant
from torch.nn import Parameter, Softmax, Softplus, ParameterDict

from utils import MaxNFEException, dirichlet_energy, rayleigh_quotient, W_dirichlet_energy
from base_classes import ODEFunc


class ODEFuncGraff(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncGraff, self).__init__(opt, data, device)
    self.in_features = in_features
    self.out_features = out_features
    self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.n_nodes = data.x.shape[0]
    self.deg_inv_sqrt = self.get_deg_inv_sqrt(data).to(device) #sending this to device because at init, data is not yet sent to device
    self.deg_inv = self.deg_inv_sqrt * self.deg_inv_sqrt
    self.data = data

    self.num_timesteps = 1
    self.time_dep_w = self.opt['time_dep_w']
    self.time_dep_struct_w = self.opt['time_dep_struct_w']
    if self.time_dep_w or self.time_dep_struct_w:
      self.num_timesteps = math.ceil(self.opt['time']/self.opt['step_size'])

    #batch norm
    if self.opt['conv_batch_norm'] == "shared":
      self.batchnorm_h = nn.BatchNorm1d(in_features)
    elif self.opt['conv_batch_norm'] == "layerwise":
      nts = math.ceil(self.opt['time'] / self.opt['step_size'])
      self.batchnorms = [nn.BatchNorm1d(in_features).to(device) for _ in range(nts)]

    #init Omega
    if self.opt['omega_style'] == 'diag':
      if self.time_dep_w:
        self.om_W = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=opt['w_param_free'])
      else:
        self.om_W = Parameter(torch.Tensor(in_features), requires_grad=opt['w_param_free'])
    elif self.opt['omega_style'] == 'zero':
      self.om_W = torch.zeros((in_features,in_features), device=device)

    #todo warning! when calcualation of symm_norm_adj moved to init (for speed) significantly reduces performance on chameleon at least, issue with computational path?
    # src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)
    # attention = torch.ones(src_deginvsqrt.shape, device=self.device)
    # self.symm_norm_adj = attention * src_deginvsqrt * dst_deginvsqrt

    #init W
    if self.opt['w_style'] in ['asymm', 'sum', 'prod', 'neg_prod']:
      if self.time_dep_w:
        self.W_dict = ParameterDict({str(i): Parameter(torch.Tensor(in_features, in_features), requires_grad=opt['w_param_free']) for i in range(self.num_timesteps)})
      else:
        self.W_W = Parameter(torch.Tensor(in_features, in_features))
    elif self.opt['w_style'] == 'diag':
      if self.opt['w_diag_init'] == 'linear':
        d = in_features
        if self.time_dep_w or self.time_dep_struct_w:
          d_range = torch.tensor([list(range(d)) for _ in range(self.num_timesteps)], device=self.device)
          self.W_D = Parameter(self.opt['w_diag_init_q'] * d_range / (d-1) + self.opt['w_diag_init_r'], requires_grad=opt['w_param_free'])
          if self.time_dep_struct_w:
            self.brt = Parameter(-2. * torch.rand((self.num_timesteps, d), device=self.device) + 1, requires_grad=True)
            self.crt = Parameter(-2. * torch.rand((self.num_timesteps, d), device=self.device) + 1, requires_grad=True)
            self.drt = Parameter(-2. * torch.rand((self.num_timesteps, d), device=self.device) + 1, requires_grad=True )
        else:
          d_range = torch.tensor(list(range(d)), device=self.device)
          self.W_D = Parameter(self.opt['w_diag_init_q'] * d_range / (d-1) + self.opt['w_diag_init_r'], requires_grad=opt['w_param_free'])
      elif self.opt['w_diag_init'] == 'binary':
          diag = torch.cat([torch.ones(in_features//2),-torch.ones(in_features - in_features//2)])
          self.W_D = Parameter(diag, requires_grad=opt['w_param_free'])
      else:
        if self.time_dep_w or self.time_dep_struct_w:
          self.W_D = Parameter(torch.ones(self.num_timesteps, in_features), requires_grad=opt['w_param_free'])
          if self.time_dep_struct_w:
            self.brt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
            self.crt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
            self.drt = Parameter(-2. * torch.rand((self.num_timesteps, in_features), device=self.device) + 1, requires_grad=True)
        else:
          self.W_D = Parameter(torch.ones(in_features), requires_grad=opt['w_param_free'])
    elif self.opt['w_style'] == 'diag_dom':
      self.W_W = Parameter(torch.Tensor(in_features, in_features - 1), requires_grad=opt['w_param_free'])
      self.t_a = Parameter(torch.Tensor(in_features), requires_grad=opt['w_param_free'])
      self.r_a = Parameter(torch.Tensor(in_features), requires_grad=opt['w_param_free'])
      if self.time_dep_w:
        self.t_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=opt['w_param_free'])
        self.r_a = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=opt['w_param_free'])
      if self.time_dep_struct_w:
        self.at = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
        self.bt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)
        self.gt = Parameter(torch.Tensor(self.num_timesteps, in_features), requires_grad=True)

    self.grad_flow_DE = []
    self.grad_flow_WDE = []
    self.grad_flow_RQ = []
    self.grad_flow_cos_high = []
    self.grad_flow_cos_low = []
    self.grad_flow_train_acc = []
    self.grad_flow_val_acc = []
    self.grad_flow_test_acc = []

    self.reset_parameters()

  def reset_parameters(self):
    # Omega
    if self.opt['omega_style'] == 'diag':
      uniform(self.om_W, a=-1, b=1)

    # W's
    if self.opt['w_style'] in ['asymm', 'sum','prod','neg_prod']:
        if self.time_dep_w:
          for i in range(self.num_timesteps):
            glorot(self.W_dict[str(i)])
        else:
          glorot(self.W_W)
    elif self.opt['w_style'] == 'diag':
      if self.opt['w_diag_init'] == 'uniform':
        uniform(self.W_D, a=-1, b=1)
        if self.time_dep_struct_w:
          uniform(self.brt, a=-1, b=1)
          uniform(self.crt, a=-1, b=1)
          uniform(self.drt, a=-1, b=1)
      elif self.opt['w_diag_init'] == 'identity':
        ones(self.W_D)
      elif self.opt['w_diag_init'] == 'linear':
        pass #done in init
    elif self.opt['w_style'] == 'diag_dom':
      if self.time_dep_struct_w:
        uniform(self.at, a=-1, b=1)
        uniform(self.bt, a=-1, b=1)
        uniform(self.gt, a=-1, b=1)
      if self.opt['w_diag_init'] == 'uniform':
        glorot(self.W_W)
        uniform(self.t_a, a=-1, b=1)
        uniform(self.r_a, a=-1, b=1)
      elif self.opt['w_diag_init'] == 'identity':
        zeros(self.W_W)
        constant(self.t_a, fill_value=1)
        constant(self.r_a, fill_value=1)
      elif self.opt['w_diag_init'] == 'linear':
        glorot(self.W_W)
        constant(self.t_a, fill_value=self.opt['w_diag_init_q'])
        constant(self.r_a, fill_value=self.opt['w_diag_init_r'])

  def get_deg_inv_sqrt(self, data):
    index_tensor = data.edge_index[0]
    deg = degree(index_tensor, self.n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt = deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    return deg_inv_sqrt

  def get_src_dst(self, x):
    """
    Get the values of a dense n-by-d matrix
    @param x:
    @return:
    """
    src = x[self.edge_index[0, :]]
    dst = x[self.edge_index[1, :]]
    return src, dst

  def set_Omega(self, T=None):
    if self.opt['omega_style'] == 'diag':
      if self.opt['omega_diag'] == 'free':
        #learnable parameters
        if self.time_dep_w:
          if T is None:
            T = 0
          Omega = torch.diag(self.om_W[T])
        else:
          Omega = torch.diag(self.om_W)
      elif self.opt['omega_diag'] == 'const':
        #constant parameters
        Omega = torch.diag(self.opt['omega_diag_val'] * torch.ones(self.in_features, device=self.device))
    elif self.opt['omega_style'] == 'zero':
      Omega = torch.zeros((self.in_features,self.in_features), device=self.device)
    return Omega

  def set_W(self, T=None):
    if T is None:
      T = 0

    if self.opt['w_style'] in ['prod']:
      if self.time_dep_w:
        return self.W_dict[str(T)] @ self.W_dict[str(T)].t()
      else:
        return self.W_W @ self.W_W.t()
    elif self.opt['w_style'] in ['neg_prod']:
      if self.time_dep_w:
        return -self.W_dict[str(T)] @ self.W_dict[str(T)].t()
      else:
        return -self.W_W @ self.W_W.t()
    elif self.opt['w_style'] in ['sum']:
      if self.time_dep_w:
        return self.W_dict[str(T)] + self.W_dict[str(T)].t()
      else:
        return (self.W_W + self.W_W.t()) / 2
    elif self.opt['w_style'] in ['asymm']:
        if self.time_dep_w:
            return self.W_dict[str(T)]
        else:
            return self.W_W

    elif self.opt['w_style'] == 'diag':
      if self.time_dep_w:
        if T is None:
          T = 0
        return torch.diag(self.W_D[T])
      elif self.time_dep_struct_w:
        if T is None:
          T = 0
        W = self.W_D[T]
        alpha = torch.diag(torch.exp(self.brt[T] * T + self.brt[T]))
        beta = torch.diag(torch.exp(-self.brt[T] * T - self.crt[T]) + self.drt[T])
        Wplus = torch.diag(F.relu(W))
        Wneg = torch.diag(-1. * F.relu(-W))
        return alpha @ Wplus - beta @ Wneg
      else:
        return torch.diag(self.W_D)
    elif self.opt['w_style'] == 'diag_dom':
      W_temp = torch.cat([self.W_W, torch.zeros((self.in_features, 1), device=self.device)], dim=1)
      W = torch.stack([torch.roll(W_temp[i], shifts=i+1, dims=-1) for i in range(self.in_features)])
      W = (W+W.T) / 2
      if self.time_dep_w:
        W_sum = self.t_a[T] * torch.abs(W).sum(dim=1) + self.r_a[T]
      elif self.time_dep_struct_w:
        W_sum = W + self.at[T] * F.tanh(self.bt[T] * T + self.gt[T]) * torch.eye(n=W.shape[0], m=W.shape[1], device=self.device)
      else:
         W_sum = self.t_a * torch.abs(W).sum(dim=1) + self.r_a
      Ws = W + torch.diag(W_sum)
      return Ws
    elif self.opt['w_style'] == 'identity':
      return torch.eye(self.in_features, device=self.device)


  def forward(self, t, x):
    if self.opt['track_grad_flow_switch']:
      self.grad_flow_DE.append(dirichlet_energy(self.data.edge_index, self.data.num_nodes, x, self.data.edge_attr, 'sym').detach().cpu().item())
      self.grad_flow_WDE.append(W_dirichlet_energy(x, self.data.edge_index, self.W).detach().cpu().item())
      self.grad_flow_RQ.append(rayleigh_quotient(self.data.edge_index, self.data.num_nodes, x).detach().cpu().item())
      self.grad_flow_cos_high.append(torch.cosine_similarity(self.high_evec, x, dim=0).mean().detach().cpu().item())
      self.grad_flow_cos_low.append(torch.cosine_similarity(self.low_evec, x, dim=0).mean().detach().cpu().item())

      logits = self.GNN_m2(x)
      data = self.data
      accs = []
      for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
      self.grad_flow_train_acc.append(accs[0])
      self.grad_flow_val_acc.append(accs[1])
      self.grad_flow_test_acc.append(accs[2])

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1

    if self.time_dep_w:
      T = int(t / self.opt['step_size'])
      self.W = self.set_W(T)
      self.Omega = self.set_Omega(T)
    else:
      T = 0

    xW = x @ self.W

    #todo warning! when calcualation of symm_norm_adj moved to init (for speed) significantly reduces performance on chameleon at least, issue with computational path?
    src_deginvsqrt, dst_deginvsqrt = self.get_src_dst(self.deg_inv_sqrt)
    attention = torch.ones(src_deginvsqrt.shape, device=self.device)
    symm_norm_adj = attention * src_deginvsqrt * dst_deginvsqrt
    # f = torch_sparse.spmm(self.edge_index, self.symm_norm_adj, x.shape[0], x.shape[0], xW)
    f = torch_sparse.spmm(self.edge_index, symm_norm_adj, x.shape[0], x.shape[0], xW)
    f = f - x @ self.Omega

    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    if self.opt['conv_batch_norm'] == "shared":
      f = self.batchnorm_h(f)
    elif self.opt['conv_batch_norm'] == "layerwise":
      f = self.batchnorms[T](f)

    #non-linearity
    if self.opt['pointwise_nonlin']:
      f = torch.relu(f)

    if self.opt['track_grad_flow_switch'] and math.isclose(t.item(), self.opt['time'] - self.opt['step_size'], rel_tol=1e-9, abs_tol=0.0):
      xT = x + self.opt['step_size'] * f
      self.grad_flow_DE.append(dirichlet_energy(self.data.edge_index, self.data.num_nodes, xT, self.data.edge_attr, 'sym').detach().cpu().item())
      self.grad_flow_WDE.append(W_dirichlet_energy(xT, self.data.edge_index, self.W).detach().cpu().item())
      self.grad_flow_DE.append(dirichlet_energy(self.data.edge_index, self.data.num_nodes, xT, self.data.edge_attr, 'sym').detach().cpu().item())
      self.grad_flow_RQ.append(rayleigh_quotient(self.data.edge_index, self.data.num_nodes, xT).detach().cpu().item())
      self.grad_flow_cos_high.append(torch.cosine_similarity(self.high_evec, xT, dim=1).mean().detach().cpu().item())
      self.grad_flow_cos_low.append(torch.cosine_similarity(self.low_evec, xT, dim=1).mean().detach().cpu().item())
      #log performance
      logits = self.GNN_m2(xT)
      data = self.data
      accs = []
      for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
      self.grad_flow_train_acc.append(accs[0])
      self.grad_flow_val_acc.append(accs[1])
      self.grad_flow_test_acc.append(accs[2])

    return f

def __repr__(self):
  return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'