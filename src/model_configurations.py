from function_laplacian_diffusion import LaplacianODEFunc
from block_constant import ConstantODEblock
from function_graff import ODEFuncGraff

class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass

def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'constant':
    block = ConstantODEblock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'graff':
    f = ODEFuncGraff
  elif ode_str in ['mlp', 'gcn_pyg', 'gcn_dgl', 'gcn_res_dgl', 'gat_dgl']:
    f = ODEFuncGraff #using for code homogeniety
  else:
    raise FunctionNotDefined
  return f
