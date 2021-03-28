import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mpl_toolkits import mplot3d
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
from collections import Counter

import matplotlib.pyplot as plt

# ------ general setup ----------
device = cfg_rob.device
# device = "cpu"

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "table_gauss.pkl")

do_plot = True
save_plot = True


def show_histogram(diffs):	
	_ = plt.hist(diffs, bins=np.arange(2, np.max(list(Counter(diffs).keys()))))
	
	plt.title(
		"Mean separation histogram over Test Dataset"
	)
	# plt.xticks(np.arange(2, np.max(list(Counter(diffs).keys()))))
	plt.xlabel('Mean Separation of SIgnal')
	plt.ylabel("Number of Instances in Test")
	plt.show()
	plt.clf()

def findMinSeparation(arr):	
	arr = [k.item() for k in arr]
	arr = np.nonzero(arr)[0]
	arr = sorted(arr)	
	n = len(arr)
	best_i = None

	diff = 1000    
	for i in range(n-1):		
		if arr[i+1] - arr[i] < diff: 
			best_i = i
			diff = arr[i+1] - arr[i]	
	return diff 

def findMeanSeparation(arr):	
	arr = [k.item() for k in arr]
	arr = np.nonzero(arr)[0]
	arr = sorted(arr)	
	n = len(arr)
	best_i = None
	sep = []

	diff = 1000    
	for i in range(n-1):
		sep.append(arr[i+1] - arr[i])

	return np.mean(sep)	

# ----- data prep -----
X_train, C_train, Y_train = [
	tmp.unsqueeze(-2).to(device)
	for tmp in load_dataset(config.set_params["path"], subset="train")
]

X_test, C_test, Y_test = [
	tmp.unsqueeze(-2).to(device)
	for tmp in load_dataset(config.set_params["path"], subset="test")
]

Y_test = remove_high_frequencies(Y_test, m=100)

min_diffs = []
mode = 'test'
# mode = 'train'
cohort = X_train if mode == 'train' else X_test
for i in tqdm(range(len(cohort))):
	sample = cohort[i][0].cpu()	
	min_diff = findMeanSeparation(sample)	
	min_diffs.append(min_diff)

plot = False
if plot:
	show_histogram(min_diffs)
	exit()

counter = Counter(min_diffs)
hist, bin_edges = np.histogram(min_diffs, bins=200)

# minimal_differences = np.arange(2, 14, 1)
# minimal_differences = np.arange(2, 4)
# minimal_differences = [2]

# ----- attack setup -----
# select samples
# samples = tuple(range(10))
samples = tuple(range(Y_test.shape[0]))


it = 1

noise_type = noise_gaussian

# noise_rel = torch.tensor(np.arange(0.0, 0.06, 0.01))
# noise_rel = torch.tensor([0.0, 0.1])
noise_rel = torch.tensor(np.arange(0.0, 0.6, 0.1))

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
# methods_include = ["Sparsity"]
methods_include = ["Tiramisu EE jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = []

# ----- perform attack -----

# select samples
X_0 = X_test[samples, ...]
Y_0 = Y_test[samples, ...]

# create result table
results = pd.DataFrame(columns=["name", "X_err"])
results.name = methods.index
results = results.set_index("name")


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

		s_len = X_0.shape[0]		
		results.loc[idx].X_err = torch.zeros(len(noise_rel), 4, s_len)

		for sep_idx in range(4):
			cur_samples = np.where(\
				np.logical_and(min_diffs >= bin_edges[sep_idx], \
					min_diffs < bin_edges[sep_idx+1]))[0]
			for s in tqdm(range(s_len)):
				X_0_s = X_0[s : s + 1, ...].repeat(it, *((X_0.ndim - 1) * (1,)))
				Y_0_s = Y_0[s : s + 1, ...].repeat(it, *((Y_0.ndim - 1) * (1,)))

				for idx_noise in range(len(noise_rel)):                
					noise_level = noise_rel[idx_noise] * Y_0_s.norm(
						p=2, dim=(-2, -1), keepdim=True
					)
					
					Y = noise_type(Y_0_s, noise_level)
					X = method.reconstr(Y, noise_level)                

					if method.name == "Sparsity":
						cur_X = X.reshape(X_0_s.shape)                
					else:
						cur_X = X

					try:
						results.loc[idx].X_err[idx_noise, sep_idx, s] = err_measure(
							cur_X, X_0_s
						).mean()
					except:
						import pdb; pdb.set_trace()
					

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

		err_mean = results.loc[idx].X_err[:, :].mean(dim=-1)
		err_std = results.loc[idx].X_err[:, :].std(dim=-1)
		

		ax = plt.axes(projection='3d')		
		
		# plot_X, plot_Y = np.meshgrid(noise_rel.numpy(), np.array(minimal_differences))		
		plot_X, plot_Y = np.meshgrid(noise_rel.numpy(), np.array(bin_edges[:4]))
		plot_Z = err_mean.numpy()		
		# import pdb; pdb.set_trace()
		# ax.plot_surface(plot_X, plot_Y , plot_Z, rstride=1, cstride=1,
		#                 cmap='viridis', edgecolor='none');
		# ax.plot_surface(plot_X, plot_Y , plot_Z, cmap='viridis', edgecolor='none');
		ax.contour3D(plot_X, plot_Y , np.transpose(plot_Z), 150, cmap='viridis')

		# plt.plot(
		# 	noise_rel,
		# 	err_mean,
		# 	linestyle=method.info["plt_linestyle"],
		# 	linewidth=method.info["plt_linewidth"],
		# 	marker=method.info["plt_marker"],
		# 	color=method.info["plt_color"],
		# 	label=method.info["name_disp"],
		# )
		# if idx == "L1" or idx == "UNet It jit":
		# plt.fill_between(
		# 	noise_rel,
		# 	err_mean + err_std,
		# 	err_mean - err_std,
		# 	alpha=0.10,
		# 	color=method.info["plt_color"],
		# )


	ax.set_xticks(noise_rel)
	ax.set_yticks(bin_edges[:4])
	# ax.set_zticks(np.arange(0, 1, step=0.05))
	
	ax.set_xlabel('noise percent')
	ax.set_ylabel('mean difference')
	ax.set_zlabel("rel.\\ $\\ell_2$ err. percent")
	plt.title("reconstruction rel.\\ $\\ell_2$ for $m=100$ coefficients, w.r.t Minimal Separation")
	# import pdb; pdb.set_trace()

	# plt.zlim((-0.01, 1.0))
	ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
	# ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
	plt.legend(loc="upper left", fontsize=12)

	if save_plot:
		fig.savefig(
			os.path.join(save_path, "fig_table_gauss.pdf"), bbox_inches="tight"
		)

	plt.show()
