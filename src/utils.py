"""
utility functions
"""
import os
import scipy
from scipy.stats import sem
import numpy as np
import random
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, get_laplacian, degree, homophily
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class MaxNFEException(Exception): pass


def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()


def make_norm(state):
  if isinstance(state, tuple):
    state = state[0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[1:1 + state_size]
    adj_y = aug_state[1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm

def add_labels(feat, labels, idx, num_classes, device):
    onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]  # convert mask to linear index
    onehot[idx, labels.squeeze()[idx]] = 1

    return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
    """
    when using labels as features need to split training nodes into training and prediction
    """
    if data.train_mask.dtype == torch.bool:
        idx = torch.where(data.train_mask)[0]
    else:
        idx = data.train_mask
    mask = torch.rand(idx.shape) < mask_rate
    train_label_idx = idx[mask]
    train_pred_idx = idx[~mask]
    return train_label_idx, train_pred_idx


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(coo, device=None):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm(  # yapf: disable
    data.edge_index, data.edge_attr, data.num_nodes,
    improved, opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo)


def get_rw_adj_old(data, opt):
  if opt['self_loop_weight'] > 0:
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                       fill_value=opt['self_loop_weight'])
  else:
    edge_index, edge_weight = data.edge_index, data.edge_attr
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  normed_csc = normalize(coo, norm='l1', axis=0)
  return coo2tensor(normed_csc.tocoo())


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
  return edge_index, edge_weight


def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[0][idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[1][idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)
  return edge_index

def dirichlet_energy(edge_index, n, X, edge_weight=None, norm_type=None):
  edge_index, L = get_laplacian(edge_index, edge_weight, norm_type)
  LX = torch_sparse.spmm(edge_index, L, n, n, X)
  # return torch.sum(torch.trace(X.T @ de))
  return (X * LX).sum()

def rayleigh_quotient(edge_index, n, X, edge_weight=None):
  energy = dirichlet_energy(edge_index, n, X, edge_weight, 'sym')
  rayleigh = energy / torch.pow(torch.norm(X, p="fro"), 2)
  return rayleigh

def W_dirichlet_energy(X, edge_index, W):
    src_x = X[edge_index[0, :]]
    dst_x = X[edge_index[1, :]]
    deg = degree(edge_index[0, :], X.shape[0])
    deginvsqrt = deg.pow_(-0.5)
    src_deginvsqrt = deginvsqrt[edge_index[0, :]]
    dst_deginvsqrt = deginvsqrt[edge_index[1, :]]

    fWf = torch.einsum("ij,jk,ik->i", src_x * src_deginvsqrt.unsqueeze(dim=1), W,
                            dst_x * dst_deginvsqrt.unsqueeze(dim=1)).data
    return -0.5 * fWf.sum()

@torch.no_grad()
def calc_stats(model, data, stats={}, pos_encoding=None):
    model.eval()
    feat = data.x
    num_nodes = data.num_nodes
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat, pos_encoding), []
    pred = logits.max(1)[1]

    try:
        X0 = model.encoder(feat, pos_encoding)
    except:
        X0 = feat
    try:
        XN = model.forward_XN(feat, pos_encoding)
    except:
        XN = model.forward(feat, pos_encoding)

    stats['T0_DE'] = dirichlet_energy(data.edge_index, num_nodes, X0, data.edge_attr, 'sym')
    x0r = X0 / torch.norm(X0, p='fro')
    stats['T0r_DE'] = dirichlet_energy(data.edge_index, num_nodes, x0r, data.edge_attr, 'sym')

    stats['TN_DE'] = dirichlet_energy(data.edge_index, num_nodes, XN, data.edge_attr, 'sym')
    xNr = XN / torch.norm(XN, p='fro')
    stats['TNr_DE'] = dirichlet_energy(data.edge_index, num_nodes, xNr, data.edge_attr, 'sym')

    stats['RQX0'] = rayleigh_quotient(data.edge_index, data.num_nodes, X0).detach().cpu().item()
    stats['RQXN'] = rayleigh_quotient(data.edge_index, data.num_nodes, XN).detach().cpu().item()

    enc_pred = X0.max(1)[1]
    stats['enc_pred_homophil'] = homophily(edge_index=data.edge_index, y=enc_pred)
    stats['pred_homophil'] = homophily(edge_index=data.edge_index, y=pred)
    stats['label_homophil'] = homophily(edge_index=data.edge_index, y=data.y)

    #spectral stats
    try:
        #passed W from run_GNN to allow calc once
        W = model.odeblock.odefunc.W #.detach().cpu().numpy()
        eig_val, eig_vec = torch.linalg.eigh(W)
        stats['ev_max'] = torch.max(eig_val).detach().cpu().item()
        stats['ev_min']= torch.min(eig_val).detach().cpu().item()
        stats['ev_av'] = torch.mean(eig_val).cpu().detach().cpu().item()
        stats['ev_std'] = torch.std(eig_val).cpu().detach().cpu().item()
        stats['ev_skew'] = torch.mean(torch.pow((eig_val - torch.mean(eig_val)) / torch.std(eig_val), 3.0)).cpu().detach().cpu().item()
        stats['T0_WDE'] = W_dirichlet_energy(X0, data.edge_index, W)
        stats['TN_WDE'] = W_dirichlet_energy(XN, data.edge_index, W)
    except:
        stats['ev_max'], stats['ev_min'], stats['ev_av'], stats['ev_std'], stats['T0_WDE'], stats['TN_WDE'] = 0., 0., 0., 0., 0., 0.

    return stats

def update_cum_stats(cum_stats, stats, rep):
    for k in stats.keys():
        old_val = cum_stats[k] * (rep - 1)
        cum_stats[k] += (old_val + stats[k]) / rep
    return cum_stats


def set_seed(seed: int = 42) -> None:
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)

# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes
