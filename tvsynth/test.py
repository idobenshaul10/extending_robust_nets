import numpy as np
import config  # isort:skip
import matplotlib.pyplot as plt
from data_management import load_dataset
# import config_robustness as cfg_rob  # isort:skip
# from config_robustness import methods  # isort:skip

device = "cpu"

X_test, C_test, Y_test = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="test")
]

for i in range(len(Y_test)):	
	fig1, ax1 = plt.subplots()
	ax1.plot(np.arange(len(X_test[i][0])), X_test[i][0].cpu())
	
	fig2, ax2 = plt.subplots()
	# ax2.plot(np.arange(len(Y_test[i][0])), Y_test[i][0].cpu())
	ax2.plot(np.arange(256), Y_test[i][0][:256], 'b-', np.arange(256), Y_test[i][0][256:], 'r--')
	plt.legend(('real', 'imaginary'))
	plt.show()
	plt.clf()