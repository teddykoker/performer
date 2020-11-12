from sklearn.datasets.samples_generator import make_circles
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# show circular data that cannot be linearly seperated
# then show how it can be with a polynomial kernel


x, y = make_circles(256, factor=0.3, noise=.05)
cmap="coolwarm"


plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(x[:,0], x[:,1] ,c=y, cmap=cmap)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig("circles.png", bbox_inches="tight")


# polynomial kernel
def poly(x):
    return np.stack([
       x[:, 0] ** 2,
       x[:, 1] ** 2,
       np.sqrt(2) * x[:, 0] * x[:, 1]
    ], axis=1)


plt.figure(figsize=(5, 5), dpi=300)
poly_x = poly(x)
ax = plt.subplot(projection='3d')
ax.scatter(poly_x[:, 0], poly_x[:, 1],poly_x[:, 1], c=y, cmap=cmap)
ax.grid(False)
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_zticks([0.0, 0.5, 1.0])
ax.set_xlabel("$\\varphi (x)_1 $")
ax.set_ylabel("$\\varphi (x)_2 $")
ax.set_zlabel("$\\varphi (x)_3 $")
plt.savefig("poly_circles.png", bbox_inches="tight")

