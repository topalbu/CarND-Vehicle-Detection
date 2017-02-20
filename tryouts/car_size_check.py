from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
image = mpimg.imread('test_images/test6.jpg')
fig, ax = plt.subplots()
mean = [0, 0]
cov = [[1280, 0], [0, 720]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 128*72).T
#plt.plot(x, y, 'x')
spacing = 20 # This can be your user specified spacing.
minorLocator = MultipleLocator(spacing)
# Set minor tick locations.
ax.yaxis.set_minor_locator(minorLocator)
ax.xaxis.set_minor_locator(minorLocator)
# Set grid to use minor tick locations.
ax.grid(which = 'minor',color='b', linestyle='-', linewidth=2)
ax.imshow(image)
#ax.grid(color='b', linestyle='-', linewidth=2)
#ax.set_xticklabels(np.arange(1, 100, 1));
#ax.set_yticklabels(np.arange(1, 10, 1));
plt.show()

# smarter version of the sliding window