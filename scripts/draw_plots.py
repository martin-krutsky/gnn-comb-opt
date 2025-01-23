import os
import copy

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
import numpy as np
import pandas as pd
import seaborn as sns
import torch


experiments = [
    'baseline', 'temp_linear', 'temp_logarithmic', 'temp_geometric', 'sign_ste', 'sign_sigmoid', 'min', 'lukasiewicz',
]
domains = ['MIS', 'MaxCut']
dregs = [3, 5, 10, 20, 30, 40, 50]
graph_sizes = [100]
random_seeds = 5

best_seeds_df = pd.read_csv('../best_seeds.csv')
int(best_seeds_df[
    (best_seeds_df['domain'] == 'MIS') & (best_seeds_df['experiment'] == 'baseline') & (best_seeds_df['d_regular'] == 3) & (best_seeds_df['graph_id'] == 0)
]['rnd_seed'].iloc[0])

logits_dicts, preacts_dicts = {}, {}
for exp in experiments:
    print(exp)
    for dom in domains:
        for dreg in dregs:
            logits_dict, preacts_dict = {}, {}  # e: np.array([]) for e in range(0, 100001, 1000)
            dict_map = {'logits': logits_dict, 'preacts': preacts_dict}
            for graph_id in range(20):
                rnd_seed = int(best_seeds_df[
                    (best_seeds_df['domain'] == dom) & (best_seeds_df['experiment'] == exp) & (best_seeds_df['d_regular'] == dreg) & (best_seeds_df['graph_id'] == graph_id)
                ]['rnd_seed'].iloc[0])
                for val_name in ['logits', 'preacts']:
                    curr_dir = f'perf_results/{exp}/{dom}/{dreg}/100/{graph_id}/{rnd_seed}/{val_name}'
                    for root, dirs, files in os.walk(curr_dir):
                        for file in files:
                            epoch = int(file.split('.')[0])
                            file_path = os.path.join(root, file)
                            curr_data = torch.load(file_path).numpy()
                            if epoch not in dict_map[val_name]:
                                dict_map[val_name][epoch] = curr_data
                            else:
                                dict_map[val_name][epoch] = np.concatenate((dict_map[val_name][epoch], curr_data))
            logits_dicts[f'{exp}_{dom}_{dreg}'] = logits_dict
            preacts_dicts[f'{exp}_{dom}_{dreg}'] = preacts_dict


def create_plots(experiment, values_dicts, dict_type='logits'):
    n_bins = 11
    for domain in ['MaxCut', 'MIS']:
        for dreg in [3, 5, 10, 20, 30, 40, 50]:
            dom_dreg = f'{experiment}_{domain}_{dreg}'
            keys = values_dicts[dom_dreg].keys()
            max_epoch = max(keys)  # min(max(keys), 40000)
            if dict_type == 'logits':
                min_val, max_val = 0, 1
            elif dict_type == 'preacts':
                min_val = min(min(values_dicts[dom_dreg][key]) for key in keys)
                max_val = max(max(values_dicts[dom_dreg][key]) for key in keys)
            else:
                raise ValueError('Unknown dict type')

            histograms = [np.histogram(values_dicts[dom_dreg][key], bins=n_bins, range=(min_val, max_val))[0] for key in
                          sorted(values_dicts[dom_dreg].keys())]  # range(0, max_epoch+1, 1000)]
            histograms = np.array(histograms).T[::-1]  # Shape: (n_epochs, n_bins)
            histograms = histograms / histograms.sum(axis=0, keepdims=True)

            plt.figure(figsize=(10, 5))
            my_cmap = copy.copy(plt.get_cmap("cividis"))  # copy the default cmap
            my_cmap.set_bad(my_cmap.colors[0])  # 'grey'
            plot = sns.heatmap(histograms, norm=LogNorm(), cmap=my_cmap,  # if dict_type == 'logits' else None
                               xticklabels=sorted(keys), yticklabels=np.linspace(min_val, max_val, n_bins)[::-1].round(2))  # range(0, max_epoch+1, 1000)
            if max_epoch > 30000:
                divisor = 5
            elif max_epoch > 20000:
                divisor = 3
            elif max_epoch > 10000:
                divisor = 2
            else:
                divisor = 1
            for ind, label in enumerate(plot.get_xticklabels()):
                if (ind % divisor) == 0:  # every 10th label is kept
                    label.set_visible(True)
                else:
                    label.set_visible(False)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Histogram of activation values" if dict_type == 'logits' else 'Histogram of pre-activation values',
                       fontsize=12)
            os.makedirs(f"imgs/pdf/{domain}", exist_ok=True)
            plt.savefig(f"imgs/pdf/{domain}/{domain}_{experiment}_{dreg}_{dict_type}.pdf", bbox_inches='tight')
            os.makedirs(f"imgs/png/{domain}", exist_ok=True)
            plt.savefig(f"imgs/png/{domain}/{domain}_{experiment}_{dreg}_{dict_type}.png", bbox_inches='tight')

            plt.title(f"{domain}: dreg={dreg}", fontsize=16)
            plt.show()
            plt.clf()

for exp in experiments:
    create_plots(exp, logits_dicts)
    create_plots(exp, preacts_dicts, dict_type='preacts')

