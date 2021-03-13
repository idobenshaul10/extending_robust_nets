import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib import rc

from data_management import load_dataset
from find_adversarial import err_measure_l2
from operators import noise_gaussian
# ----- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip
from config_robustness import methods  # isort:skip
from tqdm import tqdm
from operators import remove_high_frequencies

# ------ general setup ----------
device = cfg_rob.device

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "table_gauss.pkl")

do_plot = True
save_plot = True

# ----- data prep -----
X_test, C_test, Y_test = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="test")
]

# ----- attack setup -----

# select samples

samples = tuple(range(1))
# samples = tuple(range(1000))
it = 1

noise_type = noise_gaussian
noise_rel = torch.tensor([0.0, 0.05, 0.1])

err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["Tiramisu EE jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = []

# ----- perform attack -----

# select samples
X_0 = X_test[samples, ...]
Y_start = Y_test[samples, ...]

# create result table
results = pd.DataFrame(columns=["name", "X_err"])
results.name = methods.index
results = results.set_index("name")
# load existing results from file
if os.path.isfile(save_results):
    results_save = pd.read_pickle(save_results)
    for idx in results_save.index:
        if idx in results.index:
            results.loc[idx] = results_save.loc[idx]
else:
    results_save = results


# num_coefs_options = list(np.arange(80, 95, 5))
# num_coefs_options = list(np.arange(80, 115, 5))
num_coefs_options = [100, 110, 200]

# perform attacks
for (idx, method) in methods.iterrows():
    if idx not in methods_no_calc:

        s_len = X_0.shape[0]
        results.loc[idx].X_err = torch.zeros(len(num_coefs_options), s_len)

        for idx_noise, num_coefs in enumerate(num_coefs_options):            
            Y_0 = remove_high_frequencies(Y_start, m=num_coefs)

            for s in tqdm(range(s_len)):
                # print("Sample: {}/{}".format(s + 1, s_len))
                X_0_s = X_0[s : s + 1, ...].repeat(it, *((X_0.ndim - 1) * (1,)))
                Y_0_s = Y_0[s : s + 1, ...].repeat(it, *((Y_0.ndim - 1) * (1,)))

                noise_level = 0 * Y_0_s.norm(
                    p=2, dim=(-2, -1), keepdim=True
                )
                Y = noise_type(Y_0_s, noise_level)
                X = method.reconstr(Y, noise_level)            

                if method.name == "Sparsity":
                    cur_X = X.reshape(X_0_s.shape)               
                else:   
                    cur_X = X
                err_result = err_measure(cur_X, X_0_s).mean()                
                results.loc[idx].X_err[idx_noise, s] = err_result

# save results
for idx in results.index:
    results_save.loc[idx] = results.loc[idx]
os.makedirs(save_path, exist_ok=True)
results_save.to_pickle(save_results)

# ----- plotting -----

if do_plot:

    # LaTeX typesetting
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)

    # +++ visualization of table +++
    fig, ax = plt.subplots(clear=True, figsize=(5, 4), dpi=200)

    for (idx, method) in methods.iterrows():

        import pdb; pdb.set_trace()
        err_mean = results.loc[idx].X_err[:, :].mean(dim=-1)
        err_std = results.loc[idx].X_err[:, :].std(dim=-1)

        plt.plot(
            num_coefs_options,
            err_mean,
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            marker=method.info["plt_marker"],
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )
        # if idx == "L1" or idx == "UNet It jit":
        # plt.fill_between(
        #     num_coefs_options,
        #     err_mean + err_std,
        #     err_mean - err_std,
        #     alpha=0.10,
        #     color=method.info["plt_color"],
        # )

    plt.yticks(np.arange(0, 1, step=0.05))
    plt.ylim((-0.01, 1.0))
    plt.title("Rel.\\ $\\ell_2$ err. with respect to Number of Coefficients")    
    ax.set_xlabel('Num Coefficients')
    ax.set_ylabel("rel.\\ $\\ell_2$ err. percent")
    # ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
    # ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="upper left", fontsize=12)

    if save_plot:
        fig.savefig(
            os.path.join(save_path, "fig_table_gauss.pdf"), bbox_inches="tight"
        )

    plt.show()
