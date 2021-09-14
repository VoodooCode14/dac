

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


data = np.load("score_list.npy")


print(scipy.stats.ttest_rel(data[0:3, [4]], data[0:3, [3]]))
print(scipy.stats.ttest_rel(data[0:3, [3]], data[0:3, [2]]))
print(scipy.stats.ttest_rel(data[0:3, [2]], data[0:3, [1]]))
print(scipy.stats.ttest_rel(data[0:3, [1]], data[0:3, [0]]))

plt.bar(np.arange(0, 5), np.average(data[0:3, :], axis = 0), yerr = np.sqrt(np.var(data[0:3, :], axis = 0)))
plt.show(block = True)








