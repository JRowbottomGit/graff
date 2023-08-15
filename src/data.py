"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os
import shutil
import numpy as np
import wandb
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikipediaNetwork
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, add_remaining_self_loops, remove_self_loops
import networkx as nx
import matplotlib.pyplot as plt

from heterophilic import WebKB, Actor, WikipediaNetwork as WikipediaNetwork_het
from utils import ROOT_DIR, DummyDataset
from data_synth_hetero import get_pyg_syn_cora

DATA_PATH = f'{ROOT_DIR}/data'


def get_dataset(opt: dict, data_dir, use_lcc: bool = False) -> InMemoryDataset:
  # managing LCC for directed graphs, set to false when using opt['undirected'] == False
  if not opt['undirected'] and opt['dataset'] in ['texas', 'wisconsin', 'cornell', 'cornell_old', 'squirrel',
                                                  'chameleon']:
    try:
      wandb.config.update({'not_lcc': False}, allow_val_change=True)
    except:
      opt['not_lcc'] = False

  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  if ds in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(path, ds)
  elif ds in ['Computers', 'Photo']:
    dataset = Amazon(path, ds)
  elif ds == 'CoauthorCS':
    dataset = Coauthor(path, 'CS')
  elif ds in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
  elif ds in ['cornell_old']:
    #raw files from pre July 2022 https://github.com/bingzhewei/geom-gcn/commits
    #nb 1st split is much lower than average performance
    dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
    use_lcc = False
  elif ds in ['chameleon', 'squirrel']:
    if not os.path.isfile(f"{path}/{ds}/raw/out1_node_feature_label.txt"):
      # download with PYG for correct preproc and folder structure
      _ = WikipediaNetwork(root=path, name=ds, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
      new_raw = f"{path}/{ds}/raw"
      new_proc = f"{path}/{ds}/processed"
      os.makedirs(new_raw, exist_ok=True)
      os.makedirs(new_proc, exist_ok=True)
      shutil.move(f"{path}/{ds}/geom_gcn/raw/out1_node_feature_label.txt", f"{new_raw}/out1_node_feature_label.txt")
      shutil.move(f"{path}/{ds}/geom_gcn/raw/out1_graph_edges.txt", f"{new_raw}/out1_graph_edges.txt")
      shutil.rmtree(f"{path}/{ds}/geom_gcn/")
    dataset = WikipediaNetwork_het(root=path, name=ds, transform=T.NormalizeFeatures())

  elif ds == 'film':
    dataset = Actor(root=path, transform=T.NormalizeFeatures())

  elif ds == 'syn_cora':
    dataset = get_pyg_syn_cora(data_dir, opt, rep=1)
    use_lcc = False

  elif ds == 'bipartite':
    data = create_bipartite_graph(100, 100, 2)
    dataset = DummyDataset(data, 2)
    use_lcc = False
    try:
      wandb.config.update({'geom_gcn_splits': False}, allow_val_change=True)
    except:
      opt['geom_gcn_splits'] = False

  else:
    raise Exception('Unknown dataset.')

  if use_lcc:
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
      x=x_new,
      edge_index=torch.LongTensor(edges),
      y=y_new,
      train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data
  train_mask_exists = True
  try:
    dataset.data.train_mask
  except AttributeError:
    train_mask_exists = False

  if ds == 'ogbn-arxiv':
    split_idx = dataset.get_idx_split()
    ei = to_undirected(dataset.data.edge_index)
    data = Data(
    x=dataset.data.x,
    edge_index=ei,
    y=dataset.data.y,
    train_mask=split_idx['train'],
    test_mask=split_idx['test'],
    val_mask=split_idx['valid'])
    dataset.data = data
    train_mask_exists = True

  #todo this currently breaks with heterophilic datasets if you don't pass --geom_gcn_splits
  if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits']:
    dataset.data = set_train_val_test_split(
      12345,
      dataset.data,
      num_development=5000 if ds == "CoauthorCS" else 1500)

  if opt['self_loops']:
      dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
  else:
      dataset.data.edge_index, _ = remove_self_loops(dataset.data.edge_index)

  if opt['undirected']:
      dataset.data.edge_index = to_undirected(dataset.data.edge_index)

  return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.val_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data


def create_bipartite_graph(n, m, d, train_percent=0.8, val_percent=0.1):
    # Create the edges of the graph
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n, n+m)])

    # Create the node features of the graph
    x = torch.randn(n+m, d)

    # Define which nodes belong to which bipartition (0 or 1)
    node_type = torch.zeros(n+m, dtype=torch.long)
    node_type[n:] = 1

    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=node_type)

    # Randomly assign train, validation, and test masks based on percentages
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    indices = torch.randperm(num_nodes)
    train_end = int(num_nodes * train_percent)
    val_end = int(num_nodes * (train_percent + val_percent))
    train_mask[indices[:train_end]] = 1
    val_mask[indices[train_end:val_end]] = 1
    test_mask[indices[val_end:]] = 1
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def draw_bipartite_graph(data):
  # Create a NetworkX graph from the PyTorch Geometric Data object
  G = nx.Graph()
  for i in range(data.num_nodes):
    G.add_node(i, bipartite=data.y[i].item())
  for j in range(data.edge_index.size(1)):
    src, dst = data.edge_index[:, j].tolist()
    G.add_edge(src, dst)

  # Plot the bipartite graph
  pos = nx.bipartite_layout(G, [i for i in range(data.num_nodes) if data.y[i] == 0])
  nx.draw_networkx_nodes(G, pos, nodelist=[i for i in range(data.num_nodes) if data.y[i] == 0], node_color='r',
                         node_size=500)
  nx.draw_networkx_nodes(G, pos, nodelist=[i for i in range(data.num_nodes) if data.y[i] == 1], node_color='b',
                         node_size=500)
  nx.draw_networkx_edges(G, pos)
  plt.axis('off')
  plt.show()

if __name__ == '__main__':
    data = create_bipartite_graph(10, 5, 2)
    print(data)
    draw_bipartite_graph(data)