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

# ------ general setup ----------

device = cfg_rob.device

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "example_S6_gauss.pkl")

do_plot = True
save_plot = True

# ----- data prep -----
X_test, C_test, Y_test = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="test")
]

# ----- attack setup -----

# select samples
sample = 6
it = 1

noise_type = noise_gaussian

# select range relative noise
noise_min = 1e-3
noise_max = 0.35
noise_steps = 5
noise_rel_grid = torch.tensor(
    np.logspace(np.log10(noise_min), np.log10(noise_max), num=noise_steps)
).float()
# noise_rel_show = torch.tensor([0.00, 0.005, 0.02, 0.06, 0.12, 0.2, 0.3, 0.35]).float()
noise_rel_show = torch.tensor([0.005]).float()
noise_rel = (
    torch.cat([noise_rel_show, noise_rel_grid]).float().unique(sorted=True)
)
# noise_rel = noise_rel_show

print(noise_rel)

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["Sparsity", "Tiramisu EE jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = []

# ----- perform attack -----

# select samples
X_0 = X_test[sample : sample + 1, ...].repeat(it, *((X_test.ndim - 1) * (1,)))
Y_0 = Y_test[sample : sample + 1, ...].repeat(it, *((Y_test.ndim - 1) * (1,)))

# create result table and load existing results from file
results = pd.DataFrame(columns=["name", "X_err", "X", "Y"])
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

# perform attacks
for (idx, method) in methods.iterrows():
    if idx not in methods_no_calc:
        results.loc[idx].X_err = torch.zeros(len(noise_rel), X_0.shape[0])
        results.loc[idx].X = torch.zeros(
            len(noise_rel), *X_0.shape, device=torch.device("cpu")
        )
        results.loc[idx].Y = torch.zeros(
            len(noise_rel), *Y_0.shape, device=torch.device("cpu")
        )

        for idx_noise in range(len(noise_rel)):
            print(
                "Method: {}; Noise rel {}/{} (= {:1.3f})".format(
                    idx,
                    idx_noise + 1,
                    len(noise_rel),
                    noise_rel[idx_noise].item(),
                ),
                flush=True,
            )

            noise_level = noise_rel[idx_noise] * Y_0.norm(
                p=2, dim=(-2, -1), keepdim=True
            )            
            Y = noise_type(Y_0, noise_level)
            X = method.reconstr(Y, noise_level)            

            print(
                (
                    (Y - Y_0).norm(p=2, dim=(-2, -1))
                    / (Y_0).norm(p=2, dim=(-2, -1))
                ).mean()
            )

            # import pdb; pdb.set_trace()
            
            X = X.reshape(X_0.shape)
            results.loc[idx].X_err[idx_noise, ...] = err_measure(X, X_0)
            results.loc[idx].X[idx_noise, ...] = X.cpu()
            results.loc[idx].Y[idx_noise, ...] = Y.cpu()

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

    X_0 = X_0.cpu()
    Y_0 = Y_0.cpu()

    # method-wise plots
    for (idx, method) in methods.iterrows():

        # +++ reconstructions per noise level +++
        for idx_noise in range(len(noise_rel_show)):

            idx_noise_cur = torch.where(
                noise_rel == noise_rel_show[idx_noise]
            )[0]                    
            X_cur = results.loc[idx].X[idx_noise_cur, ...].squeeze(0)

            fig, ax = plt.subplots(clear=True, figsize=(5, 3), dpi=200)
            plt.plot(X_0[0, 0, ...], "--", color="black")
            plt.plot(X_cur[0, 0, ...], "-", color=method.info["plt_color"])

            plt.xlim(0, X_0.shape[-1])
            plt.ylim((X_0.min() - 0.1, X_0.max() + 0.1))
            plt.xticks([])
            plt.yticks([])
            ax.text(
                130,
                0.42,
                "rel.\\ $\\ell_2$ err.: {:.2f}\\%".format(
                    results.loc[idx].X_err[idx_noise_cur, 0].item() * 100
                ),
                fontsize=16,
            )

            axins = ax.inset_axes([0.05, 0.05, 0.23, 0.4])
            axins.plot(X_0[0, 0, ...], "--", color="black")
            axins.plot(X_cur[0, 0, ...], "-", color=method.info["plt_color"])

            axins.set_xlim(180, 205)
            axins.set_ylim(-0.9, -0.45)
            axins.set_xticks([])
            axins.set_yticks([])
            ax.indicate_inset_zoom(axins)

            if save_plot:
                fig.savefig(
                    os.path.join(
                        save_path,
                        "fig_example_S{}_gauss_".format(sample)
                        + method.info["name_save"]
                        + "_{:.0e}".format(noise_rel_show[idx_noise].item())
                        + ".png",
                    ),
                    bbox_inches="tight",
                )

            # not saved
            plt.title(
                method.info["name_disp"]
                + " for rel. noise level = {:1.3f}".format(
                    noise_rel_show[idx_noise].item()
                )
            )

    # +++ error curves for all methods +++
    fig, ax = plt.subplots(clear=True, figsize=(5, 4), dpi=200)

    for (idx, method) in methods.iterrows():

        err_mean = results.loc[idx].X_err[:, :].mean(dim=1)
        err_std = results.loc[idx].X_err[:, :].std(dim=1)

        plt.plot(
            noise_rel[noise_rel <= noise_max],
            err_mean[noise_rel <= noise_max],
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )
        if idx == "L1" or idx == "UNet It jit":
            plt.fill_between(
                noise_rel[noise_rel <= noise_max],
                err_mean[noise_rel <= noise_max]
                + err_std[noise_rel <= noise_max],
                err_mean[noise_rel <= noise_max]
                - err_std[noise_rel <= noise_max],
                alpha=0.10,
                color=method.info["plt_color"],
            )

    plt.yticks(np.arange(0, 1, step=0.05))
    plt.ylim((-0.008, 0.5))
    ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="upper left", fontsize=12)
    ax.set_xlabel('noise %')
    ax.set_ylabel("rel.\\ $\\ell_2$ err."
    

    if save_plot:
        fig.savefig(
            os.path.join(
                save_path, "fig_example_S{}_gauss_curve.pdf".format(sample)
            ),
            bbox_inches="tight",
        )

    plt.show()
