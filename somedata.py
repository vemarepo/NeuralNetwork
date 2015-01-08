import numpy as np
from pylab import plot, figure, show
x0 = np.random.randn(100)
x1 = np.random.randn(100)

y0 = np.sin(x0)*np.sin(x0) + np.cos(x1)*np.cos(x1)
y1 = np.sin(x1)*np.sin(x1) + np.cos(x0)*np.cos(x0)

fig = figure()
fig.add_subplot(211)
plot(x0,y0, ".")

fig.add_subplot(212)
plot(x1,y0, ".")

fig = figure()
plot(y1)

show()
