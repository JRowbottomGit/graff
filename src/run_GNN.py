import time
import numpy as np
import torch

from GNN import GNN
from data import get_dataset, set_train_val_test_split
from heterophilic import get_fixed_splits
from data_synth_hetero import get_pyg_syn_cora
from utils import calc_stats, set_seed, add_labels, get_label_masks, print_model_params
from graff_params import get_args, load_best_params, tf_ablation_args


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
    else:
        train_pred_idx = data.train_mask

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


def main(cmd_opt):

    opt = load_best_params(cmd_opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device
    rand_seed = np.random.randint(3, 10000)
    set_seed(rand_seed)
    opt['rand_seed'] = rand_seed
    if not opt['undirected'] and opt['dataset'] in ['texas', 'wisconsin', 'cornell', 'cornell_old', 'squirrel', 'chameleon']:
        opt['not_lcc'] = False # set to false when using opt['undirected'] = False

    dataset = get_dataset(opt, '../data', opt['not_lcc'])

    pos_encoding = None
    this_test = test
    results = []
    for rep in range(opt['num_splits']):
        print(f"rep {rep}")
        if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
            dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                                    num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
        if opt['geom_gcn_splits']:
            if opt['dataset'] == "Citeseer":
                opt['not_lcc'] = False
                dataset = get_dataset(opt, '../data', opt['not_lcc']) #geom-gcn citeseer uses splits over LCC and not_LCC so need to reload full DS each rep/split
            if opt['dataset'] == "cornell_old":
                data = get_fixed_splits(dataset.data, 'cornell', rep)
            else:
                data = get_fixed_splits(dataset.data, opt['dataset'], rep)
            dataset.data = data
        if opt['dataset'] == 'syn_cora':
            dataset = get_pyg_syn_cora("../data", opt, rep=rep+1)

        data = dataset.data.to(device)
        model = GNN(opt, dataset, device).to(device)

        parameters = [p for p in model.parameters() if p.requires_grad]
        print(opt)
        print_model_params(model)
        optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
        best_time = best_epoch = train_acc = val_acc = test_acc = 0
        if opt['patience'] is not None:
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
            print(f"Epoch: {epoch}, Runtime: {time.time() - start_time:.3f}, Loss: {loss:.3f}, "
                  f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
                  f"tmp_train: {tmp_train_acc:.4f}, tmp_val: {tmp_val_acc:.4f}, tmp_test: {tmp_test_acc:.4f}, "
                  f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")

            if np.isnan(loss):
                break
            if opt['patience'] is not None:
                if patience_count >= opt['patience']:
                    break
        print(
            f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")

        stats = calc_stats(model, data)
        RQX0, RQXN, ev_max, ev_min, ev_av, ev_std = stats['RQX0'], stats['RQXN'], stats['ev_max'], stats['ev_min'], stats['ev_av'], stats['ev_std']
        print(f"RQX0, RQXN, ev_max, ev_min, ev_av, l_std: {RQX0, RQXN, ev_max, ev_min, ev_av, ev_std}")

        if opt['num_splits'] > 1:
            results.append([test_acc, val_acc, train_acc, RQX0, RQXN, ev_max, ev_min, ev_av, ev_std])

    if opt['num_splits'] > 1:
        test_acc_mean, val_acc_mean, train_acc_mean, RQX0, RQXN, ev_max, ev_min, ev_av, ev_std = np.mean(results,
                                                                                                         axis=0)
        test_acc_mean = test_acc_mean * 100
        val_acc_mean = val_acc_mean * 100
        train_acc_mean = train_acc_mean * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

        results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                   'test_acc_std': test_acc_std,
                   'RQX0': RQX0, 'RQXN': RQXN, 'ev_max': ev_max, 'ev_min': ev_min, 'ev_av': ev_av, 'ev_std': ev_std}
    else:
        results = {'test_acc': test_acc, 'val_acc': val_acc, 'train_acc': train_acc,
                   'RQX0': RQX0, 'RQXN': RQXN, 'ev_max': ev_max, 'ev_min': ev_min, 'ev_av': ev_av, 'ev_std': ev_std}

    print(results)
    return results


if __name__ == '__main__':
    opt = get_args()
    opt = tf_ablation_args(opt)
    # if not opt['wandb_sweep']:
    #     opt = graff_run_params(opt)
    main(opt)