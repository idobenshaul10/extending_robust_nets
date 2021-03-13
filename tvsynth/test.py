import numpy as np
import config  # isort:skip
import matplotlib.pyplot as plt
from data_management import load_dataset
# import config_robustness as cfg_rob  # isort:skip
# from config_robustness import methods  # isort:skip

device = "cpu"

# X_test, C_test, Y_test = [
#     tmp.unsqueeze(-2).to(device)
#     for tmp in load_dataset(config.set_params["path"], subset="test")
# ]

# for i in range(len(Y_test)):	
# 	fig1, ax1 = plt.subplots()
# 	ax1.plot(np.arange(len(X_test[i][0])), X_test[i][0].cpu())
	
# 	fig2, ax2 = plt.subplots()
# 	# ax2.plot(np.arange(len(Y_test[i][0])), Y_test[i][0].cpu())
# 	ax2.plot(np.arange(256), Y_test[i][0][:256], 'b-', np.arange(256), Y_test[i][0][256:], 'r--')
# 	plt.legend(('real', 'imaginary'))
# 	plt.show()
# 	plt.clf()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random

def fun(x, y):
    return x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(-3.0, 3.0, 0.05)
y = np.arange(-3.0, 5.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array(fun(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

import pdb; pdb.set_trace()
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()