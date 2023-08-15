import os
import datetime
import torch
import wandb
import pandas as pd
from matplotlib import pyplot as plt

from wandb_run_GNN import main as run_model, track_grad_flow


def wandb_save_sweep_csv(entity, project, sweep_ids, save_id=None):
    api = wandb.Api(timeout=30)
    summary_list, config_list, name_list = [], [], []

    for sweep_id in sweep_ids:
        sweep_path = f"{entity}/{project}/{sweep_id}"
        sweep = api.sweep(sweep_path)
        for run in sweep.runs:
            summary_list.append(run.summary._json_dict)
            config_list.append(run.config)
            name_list.append(run.name)

    df_config = pd.DataFrame.from_dict(config_list)
    df_summary = pd.DataFrame.from_dict(summary_list)
    df = pd.concat([pd.DataFrame({'name': name_list}), df_config, df_summary], axis=1)

    #make string of sweep ids
    sweep_id_str = ''
    for sweep_id in sweep_ids:
        sweep_id_str += f'{sweep_id}_'

    save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_sweepdf.csv"
    df.to_csv(save_id)

    idxs = df.groupby(['gcn_params_idx', 'time', 'dataset'])['test_acc_mean'].idxmax()
    best_df = df.loc[idxs.values]
    best_save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_bestdf.csv"
    best_df.to_csv(best_save_id)

    return save_id, best_save_id


def wandb_load_sweep_csv(save_id, best_save_id):
    df = pd.read_csv(save_id, index_col=0)
    best_df = pd.read_csv(best_save_id, index_col=0)
    return df, best_df


def wandb_save_runs_csv(entity, project, sweep_id, run_ids, save_id=None, rep=None):
    # assert only one of sweep_id or run_ids is not None
    assert (sweep_id is None) != (run_ids is None)

    api = wandb.Api(timeout=25)
    summary_list, config_list, name_list = [], [], []

    if sweep_id is not None:
        sweep_path = f"{entity}/{project}/{sweep_id}"
        sweep = api.sweep(sweep_path)
        runs = sweep.runs

        if save_id is None:
            save_id = f"../ablations/{project}_{sweep_id}_runsdf.csv"
        artifact_dir = f"../ablations/{project}_{sweep_id}_tables/"
        os.makedirs(artifact_dir, exist_ok=True)


    else:
        runs = []
        for run_id in run_ids:
            run_path = f"{entity}/{project}/{run_id}"
            run = api.run(run_path)
            runs.append(run)

        if save_id is None:
            run_ids_str = "_".join(run_ids)
            save_id = f"../ablations/{project}_{run_ids_str}_runsdf.csv"
        artifact_dir = f"../ablations/{project}_{run_ids_str}_tables/"
        os.makedirs(artifact_dir, exist_ok=True)

    for run in runs:
        #for df
        summary_list.append(run.summary._json_dict)
        config_list.append(run.config)
        name_list.append(run.name)

    df_config = pd.DataFrame.from_dict(config_list)
    df_summary = pd.DataFrame.from_dict(summary_list)
    df = pd.concat([pd.DataFrame({'name': name_list}), df_config, df_summary], axis=1)
    df.to_csv(save_id)

    #download tables
    #loop over the columns in the summary and find the eval times
    for i, run in enumerate(runs):
        eval_times = []
        reps = []
        for col in run.summary._json_dict.keys():
        # for col in run.keys():
                if col.startswith('gf_table'):
                    col_split = col.split('_')
                    #find index of 't' in col_split if exists
                    if 't' in col_split:
                        t_idx = col_split.index('t')
                        eval_times.append(int(col_split[t_idx + 1]))
                    else:
                        time = run.config['time']
                        eval_times.append(time)
                    if 'rep' in col_split:
                        rep_idx = col_split.index('rep')
                        reps.append(int(col_split[rep_idx + 1]))
        break

    #get unique eval times and sorted ascending
    eval_times = sorted(list(set(eval_times)))
    if rep:
        reps = [rep]
    else:
        reps = sorted(list(set(reps)))

    for time in eval_times:
        #if there are multiple reps, plot all reps on the same plot
        if len(reps) > 0:
            for rep in reps:
                for i, run in enumerate(runs):
                    run_id = run.id
                    ds = run.config['dataset']
                    tab_id = f"gf_table_t_{time}_rep_{rep}"
                    artifact = api.artifact(f"graph_neural_diffusion/{project}/run-{run_id}-{tab_id}:v0")
                    table = artifact.get(tab_id)
                    tab_id_ds = f"gf_table_t_{time}_rep_{rep}_ds_{ds}"
                    artifact_filepath = os.path.join(artifact_dir, tab_id_ds)
                    data = table.data
                    columns = table.columns
                    df = pd.DataFrame(data, columns=columns)
                    df.to_csv(artifact_filepath, index=False)  # Save table to disk for future runs

                    # table.data.to_csv(artifact_filepath)  # Save table to disk for future runs
        else:
            for i, run in enumerate(runs):
                run_id = run.id
                # tab_id = f"gf_table_t_{time}"
                ds = run.config['dataset']
                try:
                    artifact = api.artifact(f"graph_neural_diffusion/{project}/run-{run_id}-gf_table_t{time}:v0")
                    table = artifact.get(f"gf_table_t{time}")
                    tab_id_ds = f"gf_table_t_{time}_ds_{ds}"
                    artifact_filepath = os.path.join(artifact_dir, tab_id_ds)
                    data = table.data
                    columns = table.columns
                    df = pd.DataFrame(data, columns=columns)
                    df.to_csv(artifact_filepath, index=False)  # Save table to disk for future runs

                except:
                    artifact = api.artifact(f"graph_neural_diffusion/{project}/run-{run_id}-gf_table:v0")
                    table = artifact.get(f"gf_table")
                    tab_id_ds = f"gf_table_t_{time}_ds_{ds}"
                    artifact_filepath = os.path.join(artifact_dir, tab_id_ds)
                    data = table.data
                    columns = table.columns
                    df = pd.DataFrame(data, columns=columns)
                    df.to_csv(artifact_filepath, index=False)  # Save table to disk for future runs

    return save_id

def wandb_load_runs_csv(save_id):
    df = pd.read_csv(save_id, index_col=0)
    return df


def get_runs_data(entity, project, sweep_id=None, run_ids=None, rep=None):
    if sweep_id is not None:
        save_id = f"../ablations/{project}_{sweep_id}_runsdf.csv"
    else:
        run_ids_str = "_".join(run_ids)
        save_id = f"../ablations/{project}_{run_ids_str}_runsdf.csv"

    try:
        df = wandb_load_runs_csv(save_id)
    except FileNotFoundError:
        save_id = wandb_save_runs_csv(entity, project, sweep_id, run_ids, rep=rep)
        df = wandb_load_runs_csv(save_id)

    runs = df.to_dict(orient="records")


    return save_id, runs


def load_table_from_disk(artifact_dir, tab_id):
    file_path = os.path.join(artifact_dir, tab_id)
    if os.path.exists(file_path):
        table = pd.read_csv(file_path)
        return table
    else:
        print(f"File not found: {file_path}")
        return None



def best_params(entity, project, sweep_ids, save=True,
                ds_order=['Texas', 'Wisconsin', 'Cornell', 'Film', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']):

    try:
        # make string of sweep ids
        sweep_id_str = ''
        for sweep_id in sweep_ids:
            sweep_id_str += f'{sweep_id}_'
        save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_sweepdf.csv"
        best_save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_bestdf.csv"

        df, best_df = wandb_load_sweep_csv(save_id, best_save_id)
    except:
        save_id, best_save_id = wandb_save_sweep_csv(entity, project, sweep_ids)
        df, best_df = wandb_load_sweep_csv(save_id, best_save_id)

    best_dict = {}
    for idx in df.groupby(['gcn_params_idx', 'dataset'])['test_acc_mean'].idxmax():
        row = df.loc[idx]
        gcn_params_idx = row['gcn_params_idx']
        dataset = row['dataset']
        time = row['time']
        decay = row['decay']
        lr = row['lr']
        best_dict[f'{dataset}{gcn_params_idx}'] = {'time': time, 'decay': decay, 'lr': lr}

    best_output_str = '{'
    for key, value in best_dict.items():
        best_output_str += f"'{key}': {value},\n"
    best_output_str += '}'
    print(best_output_str)

    #best dict for a specific time
    #filter df for time = 8
    # df = df[df['time'] == 8]
    long_best_dict = {}
    best_idxs = df.groupby(['gcn_params_idx', 'dataset'])['test_acc_mean'].idxmax()
    for idx in best_idxs:
        row = df.loc[idx]
        gcn_params_idx = row['gcn_params_idx']
        dataset = row['dataset']
        time = row['time']
        decay = row['decay']
        lr = row['lr']
        # long_best_dict[f'{dataset}{gcn_params_idx}'] = {'time': time, 'decay': decay, 'lr': lr}
        long_best_dict[f'{dataset}{gcn_params_idx}'] = {'decay': decay, 'lr': lr}
    long_best_output_str = '{'
    for key, value in long_best_dict.items():
        long_best_output_str += f"'{key}': {value},\n"
    long_best_output_str += '}'
    print(long_best_output_str)

    #round all the columns that are floats to 2dp
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = df[col].apply(lambda x: round(x, 2))
    #create column of format test_acc_mean  test_acc_std
    df['test_acc_mean_std'] = df['test_acc_mean'].astype(str) + ' ± ' + df['test_acc_std'].astype(str)

    table_columns = ['dataset', 'gcn_params_idx', 'test_acc_mean_std', 'train_acc_mean']#test_acc_mean', 'test_acc_std']
    best_table = df.loc[best_idxs][table_columns]
    df = best_table[table_columns]

    #change cornell_old to cornell
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('cornell_old', 'cornell'))
    df['dataset'] = df['dataset'].apply(lambda x: x.title())

    #create pivot with dataset as the columns and gcn_params_idx as the rows
    df_piv = df.pivot(index='gcn_params_idx', columns='dataset', values='test_acc_mean_std')
    df_piv_train = df.pivot(index='gcn_params_idx', columns='dataset', values='train_acc_mean')

    #put dataset columns in this order
    # ds_order = ['Texas', 'Wisconsin', 'Cornell', 'Film', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']
    # ds_order = ['Texas', 'Wisconsin', 'Cornell', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']

    #make row dataset the column names
    # df.columns = df.iloc[0]
    # #and remove the row dataset
    # df = df[1:]

    df_piv = df_piv[ds_order]
    df_piv_train = df_piv_train[ds_order]
    #increase rows and columns and print on one line
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    #rename gcn_params_idx {0:"GCN", 1:"Graff-lin", 2:"GraphSAGE"}

    print(df_piv)
    print(df_piv_train)
    if save:
        # make string of sweep ids
        sweep_id_str = ''
        for sweep_id in sweep_ids:
            sweep_id_str += f'{sweep_id}_'

        save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_best_params"
        df.to_csv(save_id)
        df_piv.to_csv(f'{save_id}_test.csv')
        df_piv_train.to_csv(f'{save_id}_train.csv')

    #convert to latex table
    print(df_piv.to_latex())

    return best_output_str, long_best_output_str


def make_perf_w_table(entity, project, sweep_ids, save=True,
                      ds_order=['Texas', 'Wisconsin', 'Cornell', 'Film', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']):

    try:
        # make string of sweep ids
        sweep_id_str = ''
        for sweep_id in sweep_ids:
            sweep_id_str += f'{sweep_id}_'
        save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_sweepdf.csv"
        best_save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_bestdf.csv"
        df, best_df = wandb_load_sweep_csv(save_id, best_save_id)
    except:
        save_id, best_save_id = wandb_save_sweep_csv(entity, project, sweep_ids)
        df, best_df = wandb_load_sweep_csv(save_id, best_save_id)

    homophilys = {'texas': 0.11, 'wisconsin': 0.21, 'cornell_old': 0.3, 'film': 0.22, 'squirrel': 0.22, 'chameleon': 0.23,
     'Citeseer': 0.74, 'Pubmed': 0.8, 'Cora': 0.81}
    df['homophily'] = df['dataset'].map(homophilys)

    table_columns = ['dataset', 'homophily', 'time', 'ev_max', 'ev_min', 'ev_av', 'ev_std',
                     'T0_DE', 'TN_DE', 'RQX0', 'RQXN', 'T0_WDE', 'TN_WDE',
                     'train_acc_mean', 'val_acc_mean', 'test_acc_mean', 'test_acc_std'] #'loss',


    df = df[table_columns]
    #round all the columns that are floats to 4dp
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = df[col].apply(lambda x: round(x, 2))

    #change cornell_old to cornell
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('cornell_old', 'cornell'))
    df['dataset'] = df['dataset'].apply(lambda x: x.title())

    #increase rows and columns and print on one line
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    #transpose the table
    df = df.transpose()
    #put dataset columns in this order
    # ds_order = ['Texas', 'Wisconsin', 'Cornell', 'Film', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']
    # ds_order = ['Texas', 'Wisconsin', 'Cornell', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']

    #make row dataset the column names
    df.columns = df.iloc[0]
    #and remove the row dataset
    df = df[1:]

    df = df[ds_order]
    print(df)
    #convert to latex table and remove blank spaces
    print(df.to_latex())#.replace(' ', ''))

    if save:
        #make string of sweep ids
        sweep_id_str = ''
        for sweep_id in sweep_ids:
            sweep_id_str += f'{sweep_id}_'

        #save as an excel file
        save_id =f"../ablations/{project}_{sweep_id_str[:-1]}_tab.xlsx"
        df.to_excel(save_id)


def learned_w_plot(entity, project, sweep_ids, save=True):

    try:
        # make string of sweep ids
        sweep_id_str = ''
        for sweep_id in sweep_ids:
            sweep_id_str += f'{sweep_id}_'
        save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_sweepdf.csv"
        best_save_id = f"../ablations/{project}_{sweep_id_str[:-1]}_bestdf.csv"

        df, best_df = wandb_load_sweep_csv(save_id, best_save_id)
    except:
        save_id, best_save_id = wandb_save_sweep_csv(entity, project, sweep_ids)
        df, best_df = wandb_load_sweep_csv(save_id, best_save_id)

    homophilys = {'texas': 0.11, 'wisconsin': 0.21, 'cornell_old': 0.3, 'film': 0.22, 'squirrel': 0.22, 'chameleon': 0.23,
     'Citeseer': 0.74, 'Pubmed': 0.8, 'Cora': 0.81}
    df['homophily'] = df['dataset'].map(homophilys)


    #change cornell_old to cornell
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('cornell_old', 'cornell'))
    df['dataset'] = df['dataset'].apply(lambda x: x.title())

    table_columns = ['dataset', 'time', 'decay', 'lr', 'ev_max', 'ev_min', 'ev_av', 'ev_std','homophily', 'W_evals', 'W_evecs']

    df = df[table_columns]
    df = df.sort_values(by=['homophily'])
    print(df)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.ravel()

    min_x = 0
    max_x = 0
    max_y = 0
    #loop through df and create a W_evals histogram for each dataset ordered by homophily
    for idx, (i, row) in enumerate(df.iterrows()):
        W_evals = eval(row['W_evals'])
        dataset = row['dataset']

        widths = [(W_evals['bins'][i] - W_evals['bins'][i + 1]) for i in range(len(W_evals['bins']) - 1)]
        min_width = min(widths)
        axs[idx].bar([(W_evals['bins'][i] + W_evals['bins'][i + 1]) / 2 for i in range(len(W_evals['bins']) - 1)],
                     W_evals['values'], align='center')

        #add title and number of x axis points + homophily
        # axs[idx].set_title(dataset + ": bins: " + str(len(W_evals['bins']) - 1) + " homophily: " + str(row['homophily']))
        axs[idx].set_title(dataset + " homophily: " + str(row['homophily']))
        axs[idx].set_xlabel('W_evals')
        axs[idx].set_ylabel('count')

        #find min and max x values and set all plots to the same x range
        if min(W_evals['bins']) < min_x:
            min_x = min(W_evals['bins'])
        if max(W_evals['bins']) > max_x:
            max_x = max(W_evals['bins'])
        #find max y value and set all plots to the same y range
        if max(W_evals['values']) > max_y:
            max_y = max(W_evals['values'])

    sweep_id_str = '_'.join(sweep_ids)
    #set main title
    plt.tight_layout()
    # fig.suptitle(f"{project}_{sweep_id_str[:-1]}_W_evals", fontsize=16)
    if save:
        plt.savefig(f"../ablations/{project}_{sweep_id_str[:-1]}_W_evals.pdf")
    plt.show()


def evol_plot(entity, project, sweep_id=None, run_ids=None, labels=None, rep=None, save=False, figsize=(10, 10), fs=10):
    # assert only one of sweep_id or run_ids is not None
    assert (sweep_id is None) != (run_ids is None)

    save_id, runs = get_runs_data(entity, project, sweep_id, run_ids, rep)

    if sweep_id is not None:
        artifact_dir = f"../ablations/{project}_{sweep_id}_tables/"
    else:
        run_id_str = '_'.join(run_ids)
        artifact_dir = f"../ablations/{project}_{run_id_str}_tables/"


    #loop over the columns in the summary and find the eval times
    for i, run in enumerate(runs):
        eval_times = []
        reps = []
        # for col in run.summary._json_dict.keys():
        for col in run.keys():
                if col.startswith('gf_table'):
                    col_split = col.split('_')
                    #find index of 't' in col_split if exists
                    if 't' in col_split:
                        t_idx = col_split.index('t')
                        eval_times.append(int(col_split[t_idx + 1]))
                    else:
                        # time = run.config['time']
                        time = run['time']
                        eval_times.append(time)
                    if 'rep' in col_split:
                        rep_idx = col_split.index('rep')
                        reps.append(int(col_split[rep_idx + 1]))
        break

    #get unique eval times and sorted ascending
    eval_times = sorted(list(set(eval_times)))
    if rep:
        reps = [rep]
    else:
        reps = sorted(list(set(reps)))

    for time in eval_times:
        #if there are multiple reps, plot all reps on the same plot
        if len(reps) > 0:
            for rep in reps:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                for i, run in enumerate(runs):
                    ds = run['dataset']
                    tab_id = f"gf_table_t_{time}_rep_{rep}_ds_{ds}"
                    artifact_filepath = os.path.join(artifact_dir, tab_id)

                    # Check if artifact file exists in local directory
                    if os.path.exists(artifact_filepath):
                        table = load_table_from_disk(artifact_dir, tab_id)
                        print(f"Loaded table {tab_id} from disk.")

                    # dataset = run.config['dataset']
                    dataset = run['dataset']
                    dataset = dataset if dataset != "cornell_old" else "cornell"

                    times = table['time'].tolist()#[d[0] for d in table.data]
                    rqs = table['RQ'].tolist()#[d[3] for d in table.data]

                    if labels:
                        label = labels[i] + " " + dataset + " RQ"
                    else:
                        label = dataset + " RQ"

                    lines = ax.plot(times, rqs, label=label)
                    ax.set_xlabel('time')
                    ax.set_ylabel('RQ')

                # increase font size of legend, ticks, and labels
                ax.legend(fontsize=fs)
                ax.tick_params(axis='both', which='major', labelsize=fs)
                ax.tick_params(axis='both', which='minor', labelsize=fs)
                ax.xaxis.label.set_size(fs)
                ax.yaxis.label.set_size(fs)
                ax.legend()
                plt.tight_layout()
                # fig.suptitle(f"{project}_{sweep_id}_t_{time}_rep_{rep}", fontsize=16)
                if save:
                    plt.savefig(f"../ablations/{project}_{sweep_id}_t_{time}_rep_{rep}_RQflow.pdf")
                plt.show()

        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            for i, run in enumerate(runs):
                ds = run['dataset']
                tab_id = f"gf_table_t_{time}_ds_{ds}"
                artifact_filepath = os.path.join(artifact_dir, tab_id)

                # Check if artifact file exists in local directory
                if os.path.exists(artifact_filepath):
                    table = load_table_from_disk(artifact_dir, tab_id)
                    print(f"Loaded table {tab_id} from disk.")

                # dataset = run.config['dataset']
                dataset = run['dataset']
                dataset = dataset if dataset != "cornell_old" else "cornell"

                times = table['time'].tolist()  # [d[0] for d in table.data]
                rqs = table['RQ'].tolist()  # [d[3] for d in table.data]

                if labels:
                    label = labels[i] + " " + dataset + " RQ"
                else:
                    label = dataset + " RQ"
                lines = ax.plot(times, rqs, label=label)
                ax.set_xlabel('time')
                ax.set_ylabel('RQ')

            #increase font size of legend, ticks, and labels
            ax.legend(fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.tick_params(axis='both', which='minor', labelsize=fs)
            ax.xaxis.label.set_size(fs)
            ax.yaxis.label.set_size(fs)

            ax.legend()
            plt.tight_layout()
            # fig.suptitle(f"{project}_{sweep_id}_{time}", fontsize=16)
            if save:
                plt.savefig(f"../ablations/{project}_{sweep_id}_t_{time}_rep_{rep}_RQflow.pdf")
            plt.show()


if __name__ == '__main__':

    #Table 1 - Best performance comparison of GCN and the GF models
    best_params(entity="graph_neural_diffusion", project="gcn_baslines", sweep_ids=["2xdshtgt","u5r7gc1i","30dcbr35"]) #plain GCN (not graff) baselines grid search
    best_params(entity="graph_neural_diffusion", project="graff_gcn_baselines", sweep_ids=["9bgy52hy","985u2idn"]) #best GRAFF from the grid search. GCN=0 wrong here

    # #Best params for Tables 2 & 4 - simple linear graff learned W - including film
    best_params(entity="graph_neural_diffusion", project="graff_simple_best", sweep_ids=["sixonmh2","oznw328l"])
    # #Data for Tables 2 & 4 - simple linear graff learned W - including film
    make_perf_w_table(entity="graph_neural_diffusion", project="graff_simple_best", sweep_ids=["sixonmh2", "oznw328l"])

    # #best params for GRAFF linear
    # # ds_order = ['Texas', 'Wisconsin', 'Cornell', 'Film', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']
    ds_order = ['Texas', 'Wisconsin', 'Cornell', 'Squirrel', 'Chameleon', 'Citeseer', 'Pubmed', 'Cora']
    best_params(entity="graph_neural_diffusion", project="graff_best_params_simple", sweep_ids=["dmkma7fp"], ds_order=ds_order) # best table rerun without film

    #Figure 2 long run RQ plot in the paper
    evol_plot(entity="graph_neural_diffusion", project="graff_simple_extrap_run_saved", sweep_id="2jc5qkzl", rep=3, save=True, figsize=(8,6), fs=14)

    #Figure 3 bipartite plot in the paper
    evol_plot(entity="graph_neural_diffusion", project="greed", run_ids=["uvnb5eja", "renxy8ax"], labels=["SGCN_gf","gcn"], save=True, figsize=(8,6), fs=14)

    #Figure 4 - histogram of simple linear graff learned W
    learned_w_plot(entity="graph_neural_diffusion", project="graff_simple_best", sweep_ids=["sixonmh2", "oznw328l"])
