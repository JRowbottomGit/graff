import argparse
import wandb

#best hyper-params for GRAFF in Table 3 and 5
best_params_dict_L = {
'chameleon': { 'w_style': 'diag_dom' ,'lr': 0.001411 ,'decay': 0.0004295 ,'dropout': 0.3674 ,'input_dropout': 0.4327 ,'hidden_dim': 64 ,'time': 3.194 ,'step_size': 1,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
# 'chameleon': { 'w_style': 'diag_dom' ,'lr': 0.001411 ,'decay': 0.0004295 ,'dropout': 0.3674 ,'input_dropout': 0.4327 ,'hidden_dim': 128 ,'time': 3.194 ,'step_size': 1,
#                        'conv_batch_norm': "layerwise", "add_source": False, "omega_style": "zero"},
'squirrel': { 'w_style': 'diag_dom' ,'lr': 0.0027 ,'decay': 0.0006 ,'dropout': 0.159 ,'input_dropout': 0.349 ,'hidden_dim': 128,'time': 3.275 ,'step_size': 1 ,
              'conv_batch_norm': "layerwise", "add_source": False, "omega_style": "zero"},
# 'texas': { 'w_style': 'diag_dom' ,'lr': 0.004145 ,'decay': 0.03537 ,'dropout': 0.3293 ,'input_dropout': 0.3936 ,'hidden_dim': 64 ,'time': 0.5756 ,'step_size': 0.5 ,
#                        'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'texas': { 'w_style': 'diag' ,'lr': 0.0040 ,'decay': 0.0089 ,'dropout': 0.3293 ,'input_dropout': 0.3936 ,'hidden_dim': 256 ,'time': 2.985 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag", "use_mlp": True}, #Dec22_small_hetero_3
# 'wisconsin': { 'w_style': 'diag' ,'lr': 0.002908 ,'decay': 0.03178 ,'dropout': 0.3717 ,'input_dropout': 0.3674 ,'hidden_dim': 64 ,'time': 2.099 ,'step_size': 0.5,
#                        'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'wisconsin': { 'w_style': 'diag' ,'lr': 0.002908 ,'decay': 0.03178 ,'dropout': 0.3717 ,'input_dropout': 0.3674 ,'hidden_dim': 256,'time': 2.099 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag", "use_mlp": True},
'cornell': { 'w_style': 'diag' ,'lr': 0.002105 ,'decay': 0.01838 ,'dropout': 0.2978 ,'input_dropout': 0.4421 ,'hidden_dim': 64 ,'time': 2.008 ,'step_size': 1,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'cornell_old': { 'w_style': 'diag' ,'lr': 0.002105 ,'decay': 0.01838 ,'dropout': 0.2978 ,'input_dropout': 0.4421 ,'hidden_dim': 256 ,'time': 2.008 ,'step_size': 1,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag", "use_mlp": True},

'film': { 'w_style': 'diag' ,'lr': 0.002602 ,'decay': 0.01299 ,'dropout': 0.4847 ,'input_dropout': 0.4191 ,'hidden_dim': 64 ,'time': 1.541 ,'step_size': 1,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'Cora': { 'w_style': 'diag' ,'lr': 0.00261 ,'decay': 0.04125 ,'dropout': 0.3386 ,'input_dropout': 0.5294 ,'hidden_dim': 64 ,'time': 3 ,'step_size': 0.25 ,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'Citeseer': { 'w_style': 'diag' ,'lr': 0.000117 ,'decay': 0.02737 ,'dropout': 0.2224 ,'input_dropout': 0.5129 ,'hidden_dim': 64 ,'time': 2 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'Pubmed': { 'w_style': 'diag' ,'lr': 0.00394 ,'decay': 0.0003348 ,'dropout': 0.4232 ,'input_dropout': 0.412 ,'hidden_dim': 64 ,'time': 2.552 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"}}

#best hyper-params for GRAFF_NL in Table 3 and 5
best_params_dict_NL = {
'chameleon': { 'w_style': 'diag_dom' , 'lr': 0.0050 ,'decay': 0.0005 ,'dropout': 0.3577 ,'input_dropout': 0.4756 ,'hidden_dim': 64 ,'time': 3.331 ,'step_size': 0.5,
                       'conv_batch_norm': "layerwise", "add_source": False, "omega_style": "zero"},
'squirrel': { 'w_style': 'diag_dom' , 'lr': 0.0065 ,'decay': 0.0009 ,'dropout': 0.1711 ,'input_dropout': 0.3545 ,'hidden_dim': 128 ,'time': 2.871 ,'step_size': 0.5,
              'conv_batch_norm': "layerwise", "add_source": False, "omega_style": "zero"},
'texas': { 'w_style': 'diag_dom' ,'lr': 0.0042 ,'decay': 0.0175 ,'dropout': 0.2346 ,'input_dropout': 0.4037 ,'hidden_dim': 32 ,'time': 2.656 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'wisconsin': { 'w_style': 'diag_dom' , 'lr': 0.0043 ,'decay': 0.0345 ,'dropout': 0.3575 ,'input_dropout': 0.3508 ,'hidden_dim': 32 ,'time': 3.785 ,'step_size': 1 ,
                       'conv_batch_norm': "layerwise", "add_source": True, "omega_style": "diag"},
'cornell': { 'w_style': 'diag' , 'lr': 0.0049 ,'decay': 0.0431 ,'dropout': 0.3576 ,'input_dropout': 0.4365 ,'hidden_dim': 32 ,'time': 2.336 ,'step_size': 1,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'film': { 'w_style': 'diag_dom' , 'lr': 0.0049 ,'decay': 0.0163 ,'dropout': 0.3682 ,'input_dropout': 0.4223 ,'hidden_dim': 32 ,'time': 1.114 ,'step_size': 1,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'Cora': { 'w_style': 'diag_dom' , 'lr': 0.0030 ,'decay': 0.0263 ,'dropout': 0.4241 ,'input_dropout': 0.5378 ,'hidden_dim': 64 ,'time': 1.445 ,'step_size': 1,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'Citeseer': { 'w_style': 'diag' , 'lr': 0.0016 ,'decay': 0.0065 ,'dropout': 0.3846 ,'input_dropout': 0.4389 ,'hidden_dim': 64 ,'time': 2.136 ,'step_size': 0.5 ,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"},
'Pubmed': { 'w_style': 'diag' , 'lr': 0.0048 ,'decay': 0.0002 ,'dropout': 0.5292 ,'input_dropout': 0.414 ,'hidden_dim': 64 ,'time': 3.343 ,'step_size': 0.5,
                       'conv_batch_norm': "none", "add_source": True, "omega_style": "diag"}}

#best hyper-params for SGCN_GF in Table 1
simple_best_params_dict = \
    {'Citeseer0': {'time': 2, 'decay': 0.005, 'lr': 0.001},
    'Cora0': {'time': 2, 'decay': 0.005, 'lr': 0.001},
    'Pubmed0': {'time': 2, 'decay': 0.0005, 'lr': 0.005},
    'chameleon0': {'time': 8, 'decay': 0.0005, 'lr': 0.005},
    'cornell_old0': {'time': 8, 'decay': 0.05, 'lr': 0.005},
    'squirrel0': {'time': 2, 'decay': 0.0005, 'lr': 0.005},
    'texas0': {'time': 8, 'decay': 0.05, 'lr': 0.005},
    'wisconsin0': {'time': 4, 'decay': 0.005, 'lr': 0.005},
    'Citeseer1': {'time': 2, 'decay': 0.005, 'lr': 0.001},
    'Cora1': {'time': 4, 'decay': 0.05, 'lr': 0.001},
    'Pubmed1': {'time': 2, 'decay': 0.0005, 'lr': 0.005},
    'chameleon1': {'time': 4, 'decay': 0.0005, 'lr': 0.005},
    'cornell_old1': {'time': 2, 'decay': 0.05, 'lr': 0.005},
    'squirrel1': {'time': 4, 'decay': 0.0005, 'lr': 0.005},
    'texas1': {'time': 2, 'decay': 0.005, 'lr': 0.005},
    'wisconsin1': {'time': 2, 'decay': 0.05, 'lr': 0.005},
    'Citeseer2': {'time': 2, 'decay': 0.005, 'lr': 0.001},
    'Cora2': {'time': 4, 'decay': 0.05, 'lr': 0.001},
    'Pubmed2': {'time': 2, 'decay': 0.0005, 'lr': 0.005},
    'chameleon2': {'time': 4, 'decay': 0.0005, 'lr': 0.005},
    'cornell_old2': {'time': 2, 'decay': 0.05, 'lr': 0.005},
    'squirrel2': {'time': 4, 'decay': 0.0005, 'lr': 0.005},
    'texas2': {'time': 2, 'decay': 0.005, 'lr': 0.005},
    'wisconsin2': {'time': 2, 'decay': 0.05, 'lr': 0.005},
    'film0': {'time': 8, 'decay': 0.005, 'lr': 0.005},
    'film1': {'time': 2, 'decay': 0.005, 'lr': 0.001},
    'film2': {'time': 2, 'decay': 0.0005, 'lr': 0.005},
    }


#best hyper-params for SGCN_GF in Figure 2
simple_long_best_params_dict = \
    {'Citeseer0': {'decay': 0.0005, 'lr': 0.005},
     'Cora0': {'decay': 0.005, 'lr': 0.001},
     'Pubmed0': {'decay': 0.0005, 'lr': 0.005},
     'chameleon0': {'decay': 0.0005, 'lr': 0.005},
     'cornell_old0': {'decay': 0.05, 'lr': 0.005},
     'squirrel0': {'decay': 0.0005, 'lr': 0.005},
     'texas0': {'decay': 0.05, 'lr': 0.005},
     'wisconsin0': {'decay': 0.05, 'lr': 0.005},
     'Citeseer1': {'decay': 0.05, 'lr': 0.001},
     'Cora1': {'decay': 0.05, 'lr': 0.005},
     'Pubmed1': {'decay': 0.0005, 'lr': 0.005},
     'chameleon1': {'decay': 0.0005, 'lr': 0.005},
     'cornell_old1': {'decay': 0.005, 'lr': 0.005},
     'squirrel1': {'decay': 0.0005, 'lr': 0.005},
     'texas1': {'decay': 0.0005, 'lr': 0.001},
     'wisconsin1': {'decay': 0.005, 'lr': 0.005},
     'Citeseer2': {'decay': 0.05, 'lr': 0.005},
     'Cora2': {'decay': 0.05, 'lr': 0.001},
     'Pubmed2': {'decay': 0.0005, 'lr': 0.005},
     'chameleon2': {'decay': 0.0005, 'lr': 0.005},
     'cornell_old2': {'decay': 0.05, 'lr': 0.005},
     'squirrel2': {'decay': 0.0005, 'lr': 0.005},
     'texas2': {'decay': 0.0005, 'lr': 0.005},
     'wisconsin2': {'decay': 0.05, 'lr': 0.005},
     'film0': {'decay': 0.005, 'lr': 0.005},
     'film1': {'decay': 0.05, 'lr': 0.005},
     'film2': {'decay': 0.005, 'lr': 0.005}
     }


def load_best_params(cmd_opt):
    if cmd_opt['use_best_params'] in ['best_lin','best_nonlin', 'simple_lin', 'simple_long']:

        if cmd_opt['use_best_params'] == 'best_lin':
            best_opt = best_params_dict_L[cmd_opt['dataset']]

        elif cmd_opt['use_best_params'] == 'best_nonlin':
            best_opt = best_params_dict_NL[cmd_opt['dataset']]

        elif cmd_opt['use_best_params'] == 'simple_lin':
            #wandb config sends nargs='+' as a list of 1 string representing the list until wandb init
            try:
                graff_gcn_params = eval(cmd_opt['graff_gcn_params'][0])
            except:
                graff_gcn_params = cmd_opt['graff_gcn_params']

            gcn_type = graff_gcn_params[0] if graff_gcn_params else cmd_opt['gcn_params_idx']
            best_opt = simple_best_params_dict[cmd_opt['dataset'] + str(gcn_type)]

        elif cmd_opt['use_best_params'] == 'simple_long':
            #wandb config sends nargs='+' as a list of 1 string representing the list until wandb init
            try:
                graff_gcn_params = eval(cmd_opt['graff_gcn_params'][0])
            except:
                graff_gcn_params = cmd_opt['graff_gcn_params']

            gcn_type = graff_gcn_params[0] if graff_gcn_params else cmd_opt['gcn_params_idx']
            best_opt = simple_long_best_params_dict[cmd_opt['dataset'] + str(gcn_type)]

        opt = {**cmd_opt, **best_opt}

    else:
        opt = cmd_opt

    return opt


def unpack_graff_params(opt):
    'unpack function for SGCN models'
    wandb.config.update({'w_style': opt['graff_params'][0]}, allow_val_change=True)
    wandb.config.update({'w_diag_init': opt['graff_params'][1]}, allow_val_change=True)
    wandb.config.update({'w_param_free': opt['graff_params'][2]}, allow_val_change=True)
    wandb.config.update({'omega_style': opt['graff_params'][3]}, allow_val_change=True)
    wandb.config.update({'omega_diag': opt['graff_params'][4]}, allow_val_change=True)
    wandb.config.update({'use_mlp': opt['graff_params'][5]}, allow_val_change=True)
    wandb.config.update({'add_source': opt['graff_params'][6]}, allow_val_change=True)

def unpack_gcn_params(opt):
    'temp function to help ablation'
    wandb.config.update({'gcn_params_idx': opt['gcn_params'][0]}, allow_val_change=True)
    wandb.config.update({'function': opt['gcn_params'][1]}, allow_val_change=True)
    wandb.config.update({'gcn_enc_dec': opt['gcn_params'][2]}, allow_val_change=True)
    wandb.config.update({'gcn_fixed': opt['gcn_params'][3]}, allow_val_change=True)
    wandb.config.update({'gcn_symm': opt['gcn_params'][4]}, allow_val_change=True)
    wandb.config.update({'gcn_non_lin': opt['gcn_params'][5]}, allow_val_change=True)


def unpack_graff_gcn_params(opt):
    'temp function to help ablation'
    wandb.config.update({'gcn_params_idx': opt['graff_gcn_params'][0]}, allow_val_change=True)
    wandb.config.update({'omega_style': opt['graff_gcn_params'][1]}, allow_val_change=True) #to make non residual
    wandb.config.update({'omega_diag': opt['graff_gcn_params'][2]}, allow_val_change=True) #choose constant or free
    wandb.config.update({'omega_diag_val': opt['graff_gcn_params'][3]}, allow_val_change=True) #choose value of diag
    wandb.config.update({'time_dep_w': opt['graff_gcn_params'][4]}, allow_val_change=True) #to share w
    wandb.config.update({'w_style': opt['graff_gcn_params'][5]}, allow_val_change=True)
    wandb.config.update({'pointwise_nonlin': opt['graff_gcn_params'][6]}, allow_val_change=True) #control nonlin
    wandb.config.update({'XN_activation': opt['graff_gcn_params'][7]}, allow_val_change=True) #don't need if using pointwise_nonlin


def graff_run_params(opt):
    'adhoc parameter overrides for running graff'
    opt['dataset'] = 'chameleon' #'bipartite'#'Cora'#texas'#'wisconsin'#'cornell_old'#'wisconsin'#'texas'
    opt['use_best_params'] = None #'simple_lin' #'best_lin','best_nonlin', 'simple_lin', 'simple_long' None#True #False
    opt['geom_gcn_splits'] = True
    opt['num_splits'] = 1

    opt['w_style'] = 'sum' #'diag_dom' #'neg_prod'#diag_dom'
    if opt['w_style'] == 'diag_dom':
        # opt['graff_params'] = ['diag_dom', 'uniform', True, 'diag', 'free', False, True]#, True]
        opt['w_style'] = 'diag_dom'
        opt['w_diag_init'] = 'uniform'
        opt['w_param_free'] = True #set to False for Squirrel?
        opt['omega_style'] = 'diag'
        opt['omega_diag'] = 'free'
        opt['use_mlp'] = False
        opt['add_source'] = True
    elif opt['w_style'] == 'diag':
        # opt['graff_params'] = ['diag', 'uniform', False, 'diag', 'free', True, True]#, True]
        opt['w_style'] = 'diag'
        opt['w_diag_init'] = 'uniform'
        opt['w_param_free'] = False
        opt['omega_style'] = 'diag'
        opt['omega_diag'] = 'free'
        opt['use_mlp'] = True
        opt['add_source'] = True

    opt['block'] = 'constant'
    opt['function'] = 'graff' #gcn_res_dgl'#'gcn_dgl'#'gcn_pyg'#'mlp'#'graff'
    # opt['graff_gcn_params'] = [1, 'zero', 'none', 0, False, 'sum', False, True] #linear graff
    # opt['graff_gcn_params'] = [0, 'diag', 'const', 1, True, 'asymm', True, False] #gcn
    # opt['gcn_params'] = [5, 'gcn_res_dgl', True, True, True, False]
    # opt['time'] = 8

    opt['XN_activation'] = True
    opt['conv_batch_norm'] = 'layerwise'#False #"layerwise"
    opt['pointwise_nonlin'] = False

    opt['optimizer'] = 'adam'
    opt['epoch'] = 25
    opt['method'] = 'euler'
    opt['self_loops'] = True #False
    opt['undirected'] = True #False

    opt['wandb'] = True
    opt['track_grad_flow'] = True
    opt['track_grad_flow_switch'] = False
    return opt


def t_or_f(tf_str):
    if tf_str == "True" or tf_str == "true" or (type(tf_str) == bool and tf_str):
        return True
    elif tf_str == "False" or tf_str == "false" or (type(tf_str) == bool and not tf_str):
        return False
    else:
        return tf_str

def tf_ablation_args(opt):
    for arg in list(opt.keys()):
        str_tf = opt[arg]
        bool_tf = t_or_f(str_tf)
        opt[arg] = bool_tf
    return opt


def get_args():
    parser = argparse.ArgumentParser()
    #run args
    parser.add_argument('--rand_seed', type=int, default=42, help='tracking rand seed for reproducibility')
    parser.add_argument('--use_best_params', type=str, default='best_lin', help=['best_lin','best_nonlin', 'simple_lin', 'simple_long', 'none'])
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--epoch', type=int, default=150, help='Number of training epochs per iteration.')
    parser.add_argument('--patience', type=int, default=None, help='set if training should use patience on val acc')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')

    # data args
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora, Citeseer, Pubmed, texas, wisconsin, cornell_old, chameleon, squirrel, bipartite')
    parser.add_argument('--data_norm', type=str, default='rw', help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5, help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true', help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true', help='use the 10 fixed splits from https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1, help='the number of splits to repeat the results on')
    parser.add_argument('--not_lcc', action="store_false", help="don't use the largest connected component")
    # parser.add_argument('--self_loops', action='store_false', help='control self loops')
    # parser.add_argument('--undirected', action='store_false', help='control undirected')
    parser.add_argument('--target_homoph', type=str, default='0.80', help='target_homoph for syn_cora [0.00,0.10,..,1.00]')

    # GNN args
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='graff', help='laplacian, transformer, greed, GAT')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true', help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    # parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true', help='Add a fully connected layer to the encoder.')
    # parser.add_argument('--add_source', dest='add_source', action='store_true', help='beta*x0 source term')
    # parser.add_argument('--XN_activation', action='store_true', help='whether to relu activate the terminal state')
    # parser.add_argument('--m2_mlp', action='store_true', help='whether to use decoder mlp')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true', help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, default='euler', help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=0.1, help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun", help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true', help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0, help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000, help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--max_test_steps", type=int, default=100, help="Maximum number steps for the dopri5Early test integrator. used if getting OOM errors at test time")

    # graff args
    parser.add_argument('--omega_style', type=str, default='zero', help='zero, diag')
    parser.add_argument('--omega_diag', type=str, default='free', help='free, const')
    parser.add_argument('--omega_params', nargs='+', default=None, help='list of Omega args for ablation')
    parser.add_argument('--w_style', type=str, default='sum', help='sum, prod, neg_prod, diag_dom, diag')
    parser.add_argument('--w_diag_init', type=str, default='uniform', help='init of diag elements [identity, uniform, linear]')
    # parser.add_argument('--w_param_free', action='store_false', help='allow parameter to require gradient')
    parser.add_argument('--w_diag_init_q', type=float, default=1.0, help='slope of init of spectrum of W')
    parser.add_argument('--w_diag_init_r', type=float, default=0.0, help='intercept of init of spectrum of W')
    parser.add_argument('--time_dep_w', action='store_true', help='Learn a time dependent potentials')
    parser.add_argument('--time_dep_struct_w', action='store_true', help='Learn a structured time dependent potentials')
    parser.add_argument('--conv_batch_norm', type=str, default='', help='layerwise, shared')
    # parser.add_argument('--pointwise_nonlin', action='store_true', help='apply pointwise nonlin relu to f')
    parser.add_argument('--graff_params', nargs='+', default=None, help='list of args for focus models')

    # wandb and raytune logging and tuning
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_offline', action='store_true')  # https://docs.wandb.ai/guides/technical-faq
    parser.add_argument('--wandb_sweep', action='store_true', help="flag if sweeping")
    parser.add_argument('--wandb_entity', default="graph_neural_diffusion", type=str, help="jrowbottomwnb")
    parser.add_argument('--wandb_project', default="greed", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--track_grad_flow', action='store_true', help="flag to track DE and RQ in evol")
    parser.add_argument('--track_grad_flow_switch', action='store_true', help="switch to track DE and RQ in evol")
    parser.add_argument('--raytune', action='store_true', help="flag if logging to raytune")
    parser.add_argument('--torch_save_model', action='store_true', help="save model as torch save")
    parser.add_argument('--torch_load_track_gf', action='store_true', help="load model for track grad flow")

    # GCN ablation args
    parser.add_argument('--gcn_fixed', type=str, default='False', help='fixes layers in gcn')
    parser.add_argument('--gcn_enc_dec', type=str, default='False', help='uses encoder decoder with GCN')
    parser.add_argument('--gcn_non_lin', type=str, default='False', help='uses non linearity with GCN')
    parser.add_argument('--gcn_symm', type=str, default='False', help='make weight matrix in GCN symmetric')
    parser.add_argument('--gcn_bias', type=str, default='False', help='make GCN include bias')
    parser.add_argument('--gcn_mid_dropout', type=str, default='False', help='dropout between GCN layers')
    parser.add_argument('--gcn_params', nargs='+', default=None, help='list of args for gcn ablation')
    parser.add_argument('--gcn_params_idx', type=int, default=None, help='index to track GCN ablation')

    parser.add_argument('--graff_gcn_params', nargs='+', default=None, help='list of args for gcn ablation')

    # workaround for https://github.com/wandb/wandb/issues/1700
    # Current True/False args for wandb sweeps greyed out above
    parser.add_argument('--self_loops', type=str, default="False", help='control self loops')
    parser.add_argument('--undirected', type=str, default="True", help='control undirected')
    parser.add_argument("--batch_norm", type=str, default="False", help='search over reg params')
    parser.add_argument('--use_mlp', type=str, default="False", help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', type=str, default="False", help='beta*x0 source term')
    parser.add_argument('--XN_activation', type=str, default="True", help='whether to relu activate the terminal state')
    parser.add_argument('--m2_mlp', type=str, default="False", help='whether to use decoder mlp')
    parser.add_argument('--w_param_free', type=str, default="True", help='allow parameter to require gradient')
    parser.add_argument('--pointwise_nonlin', type=str, default="False", help='apply pointwise nonlin relu to f')

    args = parser.parse_args()
    opt = vars(args)
    return opt