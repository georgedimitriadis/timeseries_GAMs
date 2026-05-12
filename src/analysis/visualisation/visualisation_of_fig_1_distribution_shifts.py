
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)
X, Y = np.meshgrid(x, y)

Z = np.cos(X)*np.cos(Y)

subplot = plt.subplot2grid(shape=(1,1), loc=(0,0), colspan=1, rowspan=1, projection='3d')
surf = subplot.plot_surface(X, Y, Z, linewidth=1, antialiased=False, cmap='coolwarm')

fig = plt.figure()
subplot = plt.subplot2grid(shape=(1,1), loc=(0,0), colspan=1, rowspan=1, fig=fig)
subplot.pcolor(X, Y, Z, cmap='rainbow')


x = np.linspace(0, 10, 50)
c = np.linspace(-1, 1, 10)
X, C = np.meshgrid(x, c)
Y = np.cos(X)

eps = 0.1
for i in range(10):
    for j in range(50):
        if not C[i, j] - eps < Y[i, j] < C[i, j] + eps:
            Y[i, j] = -5

Y = np.cos(X)
eps = 0.1
for i in range(10):
    for j in range(50):
        if np.random.rand() < 0.8:
            Y[i, j] = -5

fig = plt.figure()
subplot = plt.subplot2grid(shape=(1,1), loc=(0,0), colspan=1, rowspan=1, fig=fig)
subplot.pcolor(X, C, Y, cmap='rainbow')