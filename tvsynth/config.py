import os

from data_management import sample_tv_signal
from operators import Gaussian, Fourier


DATA_PATH = os.path.join("tv_synth_raw_data_fourier_dense")
RESULTS_PATH = os.path.join("tv_synth_results_fourier_dense")

# ----- random seeds -----
torch_seed = 1
numpy_seed = 2
matrix_seed = 3

# ----- signal configuration -----
n = 256  # signal dimension
data_params = {  # additional data generation parameters
    "j_min": 5,
    "j_max": 20,
    "min_dist": 8,
    "bound": 10,
    "min_height": 0.5,
}
data_gen = sample_tv_signal  # data generator function

# ----- measurement configuration -----
# m = 100  # measurement dimension
m = n
meas_params = {"seed": matrix_seed}  # additional measurement parameters
meas_op = Fourier # measurement operator

set_params = {
    "num_train": 200000,
    "num_val": 1000,
    "num_test": 1000,
    "path": DATA_PATH,
}