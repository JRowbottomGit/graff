import time
import os
import numpy as np
import torch
import wandb

from GNN import GNN
from data import get_dataset, set_train_val_test_split
from heterophilic import get_fixed_splits
from data_synth_hetero import get_pyg_syn_cora
from utils import calc_stats, set_seed, add_labels, get_label_masks, print_model_params, update_cum_stats
from graff_params import get_args, load_best_params, tf_ablation_args, unpack_gcn_params, unpack_graff_params, unpack_graff_gcn_params
from GNN_GCN import GCNs, MLP


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def train(model, optimizer, data, pos_encoding=None):
    lf = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])
        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)

    out = model(feat, pos_encoding)

    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
    model.eval()
    feat = data.x
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat, pos_encoding), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def track_grad_flow(model, dataset, data, opt, L_decomp=False, rep=None, pos_encoding=None):
    if L_decomp:
        i = dataset.data.edge_index
        n = dataset.data.num_nodes
        adj = model.odeblock.odefunc.symm_norm_adj
        coo_adj = torch.sparse_coo_tensor(i, adj, (n, n))
        torch_eye = torch.eye(n)
        coo_lap = torch_eye - coo_adj
        L_evals, L_evecs = torch.linalg.eig(coo_lap)

        mag_evals = torch.abs(L_evals)
        high_eval = torch.max(mag_evals)
        low_eval = torch.min(mag_evals)
        # assuming eigenvalues of graph laplacian are real
        high_evec = L_evecs[:, torch.argmax(mag_evals)].real.unsqueeze(1)
        low_evec = L_evecs[:, torch.argmin(mag_evals)].real.unsqueeze(1)

        model.odeblock.odefunc.high_evec = high_evec
        model.odeblock.odefunc.low_evec = low_evec

    else:
        model.odeblock.odefunc.high_evec = torch.zeros((dataset.data.num_nodes,1))
        model.odeblock.odefunc.low_evec = torch.zeros((dataset.data.num_nodes,1))

    wandb.config.update({'track_grad_flow_switch': True}, allow_val_change=True)
    model.eval()
    feat = data.x
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat, pos_encoding), []
    wandb.config.update({'track_grad_flow_switch': False}, allow_val_change=True)

    grad_flow_DE = model.odeblock.odefunc.grad_flow_DE
    grad_flow_WDE = model.odeblock.odefunc.grad_flow_WDE
    grad_flow_RQ = model.odeblock.odefunc.grad_flow_RQ
    grad_flow_cos_high = model.odeblock.odefunc.grad_flow_cos_high
    grad_flow_cos_low = model.odeblock.odefunc.grad_flow_cos_low
    grad_flow_train_acc = model.odeblock.odefunc.grad_flow_train_acc
    grad_flow_val_acc = model.odeblock.odefunc.grad_flow_val_acc
    grad_flow_test_acc = model.odeblock.odefunc.grad_flow_test_acc

    if opt['wandb']:
        W = model.odeblock.odefunc.W
        W_evals, W_evecs = torch.linalg.eigh(W)
        if rep is not None:
            wandb.log({f"W_evals_t_{opt['time']}_rep_{rep}": W_evals, f"W_evecs_t_{opt['time']}_rep_{rep}": W_evecs})
        else:
            wandb.log({f"W_evals": W_evals, f"W_evecs": W_evecs})

        # https://docs.wandb.ai/guides/track/log/log-tables
        times = np.arange(0, opt['time'] + opt['step_size'], opt['step_size'])
        data = [[time, DE, WDE, RQ, CH, CL, train, val, test] for (time, DE, WDE, RQ, CH, CL, train, val, test) in
                zip(times, grad_flow_DE, grad_flow_WDE, grad_flow_RQ, grad_flow_cos_high, grad_flow_cos_low,
                    grad_flow_train_acc, grad_flow_val_acc, grad_flow_test_acc)]
        wand_table = wandb.Table(data=data,
                                 columns=["time", "DE", "WDE", "RQ", "CH", "CL", "train_acc", "val_acc", "test_acc"])
        if rep is not None:
            wandb.log({f"gf_table_t_{opt['time']}_rep_{rep}": wand_table})
        else:
            wandb.log({f"gf_table_t_{opt['time']}": wand_table})


def main(cmd_opt, data_dir="../data"):
    print(f"working dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"data dir: {data_dir}")

    opt = load_best_params(cmd_opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device
    rand_seed = np.random.randint(3, 10000)
    set_seed(rand_seed)
    opt['rand_seed'] = rand_seed

    if opt['wandb']:
        if opt['wandb_offline']:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"

        if 'wandb_run_name' in opt.keys():
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                                   name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)
        else:
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                                   reinit=True, config=opt, allow_val_change=True)

        opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print(opt['gcn_params'])
    if opt['gcn_params']: #temp function for GCN ablation
        unpack_gcn_params(opt)
    elif opt['graff_params']: #temp function for GCN ablation
        unpack_graff_params(opt)
    elif opt['graff_gcn_params']: #temp function for GCN ablation
        unpack_graff_gcn_params(opt)

    dataset = get_dataset(opt, data_dir, opt['not_lcc'])

    pos_encoding = None
    this_test = test
    run_times = []
    results = []
    cum_stats = {}
    for rep in range(opt['num_splits']):
        print(f"rep {rep}")
        if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
            dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                                    num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
        if opt['geom_gcn_splits']:
            if opt['dataset'] == "Citeseer":
                wandb.config.update({'not_lcc': False}, allow_val_change=True)     # opt['not_lcc'] = False
                dataset = get_dataset(opt, data_dir, opt['not_lcc'])       #geom-gcn citeseer uses splits over LCC and not_LCC so need to reload full DS each rep/split

            if opt['dataset'] == "cornell_old":
                data = get_fixed_splits(dataset.data, 'cornell', rep)
            else:
                data = get_fixed_splits(dataset.data, opt['dataset'], rep)

            dataset.data = data
        if opt['dataset'] == 'syn_cora':
            dataset = get_pyg_syn_cora(data_dir, opt, rep=rep+1)

        data = dataset.data.to(device)
        if opt['function'] in ['mlp']:
            model = MLP(opt, dataset, device=device).to(device)
        elif opt['function'] in ['gcn_pyg', 'gcn_dgl', 'gcn_res_dgl', 'gat_dgl']:
            hidden_feat_repr_dims = int(opt['time'] // opt['step_size']) * [opt['hidden_dim']]
            feat_repr_dims = [dataset.data.x.shape[1]] + hidden_feat_repr_dims + [dataset.num_classes]
            model = GCNs(opt, dataset, device, feat_repr_dims, dropout=opt['dropout']).to(device)
        else:
            model = GNN(opt, dataset, device).to(device)

        parameters = [p for p in model.parameters() if p.requires_grad]
        print(opt)
        print_model_params(model)
        optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
        best_time = best_epoch = train_acc = val_acc = test_acc = 0

        patience_count = 0
        for epoch in range(1, opt['epoch']):
            start_time = time.time()
            loss = train(model, optimizer, data, pos_encoding)
            tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

            best_time = opt['time']
            if tmp_val_acc > val_acc:
                best_epoch = epoch
                train_acc = tmp_train_acc
                val_acc = tmp_val_acc
                test_acc = tmp_test_acc
                best_time = opt['time']
                patience_count = 0
            else:
                patience_count += 1
            run_time = time.time() - start_time
            run_times.append(run_time)

            print(f"Epoch: {epoch}, Runtime: {run_time:.3f}, Loss: {loss:.3f}, "
                  f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
                  f"tmp_train: {tmp_train_acc:.4f}, tmp_val: {tmp_val_acc:.4f}, tmp_test: {tmp_test_acc:.4f}, "
                  f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")

            if np.isnan(loss):
                break
            if opt['patience'] is not None:
                if patience_count >= opt['patience']:
                    break
        print(f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")

        stats = calc_stats(model, data)

        if opt['num_splits'] > 1:
            results.append([test_acc, val_acc, train_acc, loss])
            if rep == 0:
                cum_stats = stats
            else:
                cum_stats = update_cum_stats(cum_stats, stats, rep)

        if opt['torch_save_model']:
            torch.save(model.state_dict(), f"../ablations/saved_models/{opt['dataset']}_{opt['time']}_split_{rep}.pt")

        if opt['torch_load_track_gf']:
            model.load_state_dict(torch.load(f"../ablations/saved_models/{opt['dataset']}_8_split_{rep}.pt"))
            track_grad_flow(model, dataset, data, opt, rep=rep)


    if opt['num_splits'] > 1:
        test_acc_mean, val_acc_mean, train_acc_mean, loss = np.mean(results, axis=0) * 100 #RQX0_mean, RQXN_mean, ev_max_mean, ev_min_mean, ev_av_mean, ev_std_mean\
        loss = loss / 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
        results_dict = {'test_acc_mean': test_acc_mean, 'val_acc_mean': val_acc_mean, 'train_acc_mean': train_acc_mean, 'test_acc_std': test_acc_std, 'loss_mean': loss}
        all_stats = {**results_dict, **cum_stats}
    else:
        results_dict = {'test_acc': test_acc, 'val_acc': val_acc, 'train_acc': train_acc, 'loss': loss} #,'RQX0': RQX0, 'RQXN': RQXN, 'ev_max': ev_max, 'ev_min': ev_min, 'ev_av': ev_av, 'ev_std': ev_std}
        all_stats = {**results_dict, **stats}

    all_stats['num_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_stats['av_fwd'] = np.mean(run_times)
    all_stats['std_fwd'] = np.std(run_times)

    #post training analysis of the last split
    if opt['track_grad_flow']:
        track_grad_flow(model, dataset, data, opt, pos_encoding=None)
        # wandb.config.update({'time': 64}, allow_val_change=True)  # opt['not_lcc'] = False
        # track_grad_flow(model, dataset, data, opt, pos_encoding=None)

    if opt['wandb']:
        wandb.log(all_stats)
        wandb_run.finish()

    print(all_stats)

    return all_stats

if __name__ == '__main__':
    opt = get_args()
    opt = tf_ablation_args(opt)
    # if not opt['wandb_sweep']:
    #     opt = graff_run_params(opt)
    main(opt)

    # terminal commands to run sweeps
    # wandb sweep ../wandb_sweep_configs/<filename>.yaml
    # ./run_sweeps.sh XXX
    # nohup ./run_sweeps.sh XXX &

    #wandb commands
    #wandb agent graph_neural_diffusion/gcn_baslines_cora/d4icv0n2