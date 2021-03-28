
# Solving Inverse Problems With Deep Neural Networks - Robustness Included?

[![GitHub license](https://img.shields.io/github/license/jmaces/robust-nets)](https://github.com/jmaces/robust-nets/blob/master/LICENSE)
[![code-style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-Pytorch-1f425f.svg)](https://pytorch.org/)

This repository is based on the official implementation of the paper [Solving Inverse Problems With Deep Neural Networks - Robustness Included?](http://arxiv.org/abs/2011.04268) by M. Genzel, J. Macdonald, and M. März (2020).

## Content

This repository contains subfolders for five experimental scenarios. Each of them is independent of the others.

- [`tvsynth`](tvsynth) : Signal recovery of piecewise constant 1D signals (following a total variation synthesis model) from random Gaussian measurements.

## Requirements

The package versions are the ones we used. Other versions might work as well.

`matplotlib` *(v3.1.3)*  
`numpy` *(v1.18.1)*  
`pandas` *(v1.0.5)*  
`piq` *(v0.4.1)*  
`python` *(v3.8.3)*  
`pytorch` *(v1.4.0)*  
`scikit-image` *(v0.16.2)*  
`torchvision` *(v0.5.0)*  
`tqdm` *(v4.46.0)*

## Usage

Each of the individual experiment subfolders contains configuration files as well
as scripts for preparing the data, for training the neural networks, for obtaining total variation minimization reconstructions, and for finding adversarial perturbations.

The details are described within in each subfolder.

## Acknowledgements
Our implementation of the Tiramisu network is based on and adapted from https://github.com/bfortuner/pytorch_tiramisu/.  

## License

This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
Presentation: https://docs.google.com/presentation/d/1xfYK_Kv4vwkUZxDh_MyfT5pbCpitOG_kuGZERT8BQkw/edit?usp=sharing

Running Commands:
`cd tvsynth/`

 - To create the data, change the configs in `config.py` and run:
 `python data_management.py`
	 -  The dataset is based on the following parameters: "j_min", "j_max", "min_dist", "bound", "min_height" as given in presentation.    
 - To train network on data, change the DATA and RESULTS folder in `config.py`, and run:
 `python script_train_tiramisu_ee_jitter.py`. Other networks train scripts are also given. 
 - To run robustness on Gaussian Noise across entire dataset, configure the parameters in `config_robustness.py` and run:  `python script_robustness_table_gauss.py`.
 - To run on a single test example, run: `python script_robustness_example_gauss.py`
 - To compare minimal/mean separation across Gaussian Noise Levels, use the script: `script_robustness_example_gauss_minimal_separation.py`
 




